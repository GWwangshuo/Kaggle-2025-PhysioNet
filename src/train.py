import argparse
import datetime
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import src.utils.misc as misc
from src.models.unet import UNet

from engine import train_one_epoch, evaluate
from src.utils.transforms import build_transforms
from src.utils.datasets import ECGScanDataset
from configs.default import cfg

os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'


def get_args_parser():
    parser = argparse.ArgumentParser('PhysioNet', add_help=False)

    # training
    parser.add_argument('--start_epoch', default=cfg.START_EPOCH, type=int, metavar='N',
                        help='Starting epoch')
    parser.add_argument('--epochs', default=cfg.EPOCHS, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=cfg.WARMUP_EPOCHS, metavar='N',
                        help='Epochs to warm up LR')
    parser.add_argument('--batch_size', default=cfg.BATCH_SIZE, type=int,
                        help='Batch size per GPU (effective batch size = batch_size * # GPUs)')
    parser.add_argument('--lr', type=float, default=cfg.LR, metavar='LR',
                        help='Learning rate (absolute)')
    parser.add_argument('--min_lr', type=float, default=cfg.MIN_LR, metavar='LR',
                        help='Minimum LR for cyclic schedulers that hit 0')
    parser.add_argument('--lr_schedule', type=str, default=cfg.LR_SCHEDULE,
                        help='Learning rate schedule')
    parser.add_argument('--weight_decay', type=float, default=cfg.WEIGHT_DECAY,
                        help='Weight decay (default: 0.0)')

    parser.add_argument('--seed', default=cfg.SEED, type=int)
    parser.add_argument('--num_workers', default=cfg.NUM_WORKERS, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for faster GPU transfers')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # dataset
    parser.add_argument('--data_path', default=cfg.DATA_PATH, type=str,
                        help='Path to the dataset')
    parser.add_argument('--second_stage_data_path', default=cfg.SECOND_STAGE_DATA_PATH, type=str,
                        help='Path to the second stage dataset')
    parser.add_argument('--fold', default=cfg.FOLD, type=str,
                        help='Whether to use full dataset or part to train')

    # checkpointing
    parser.add_argument('--pretrained_weights', default=cfg.PRETRAINED_WEIGHTS,
                        help='Pretrained weights of model')
    parser.add_argument('--output_dir', default=cfg.OUTPUT_DIR,
                        help='Directory to save outputs (empty for no saving)')
    parser.add_argument('--save_last_freq', type=int, default=cfg.SAVE_LAST_FREQ,
                        help='Frequency (in epochs) to save checkpoints')
    parser.add_argument('--log_freq', default=cfg.LOG_FREQ, type=int)
    parser.add_argument('--device', default=cfg.DEVICE,
                        help='Device to use for training/testing')

    # distributed training
    parser.add_argument('--world_size', default=1, type=int,
                        help='Number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL used to set up distributed training')

    return parser

def freeze_baseline(model):
    """
    Freeze all parameters except cross-lead and refine head.
    """
    for name, p in model.named_parameters():
        if "y_offset_head" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False

def load_model(**kwargs):
    weights_path = kwargs.pop("weights_path", None)
    model = UNet(**kwargs)
    state = torch.load(weights_path, map_location='cpu')
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    verbose = model.load_state_dict(state, strict=False)
    print(verbose)
    model.eval()
    return model

def main(args):
    misc.init_distributed_mode(args)
    print('Job directory:', os.path.dirname(os.path.realpath(__file__)))
    print("Arguments:\n{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Set seeds for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    # Set up TensorBoard logging (only on main process)
    if global_rank == 0 and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # Data augmentation transforms
    transform_train = build_transforms()

    dataset_train = ECGScanDataset(
        raw_data_path=args.data_path,
        second_stage_data_path=args.second_stage_data_path,
        transform=transform_train,
        mode='train'
    )
    print(dataset_train)
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train =", sampler_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    
    if args.fold != 'all':
        dataset_valid = ECGScanDataset(
            raw_data_path=args.data_path,
            second_stage_data_path=args.second_stage_data_path,
            transform=transform_train,
            mode='valid'
        )
        print(dataset_valid)
    
        sampler_valid = torch.utils.data.DistributedSampler(
            dataset_valid, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        print("Sampler_valid =", sampler_valid)
        
        data_loader_valid = torch.utils.data.DataLoader(
            dataset_valid,
            sampler=sampler_valid,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    
    model = load_model(
        weights_path=args.pretrained_weights, # 使用从 args 获取的路径
        **cfg.MODEL_PARAMS                    # 解包模型架构参数
    )

    print("Model =", model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    model.to(device)

    eff_batch_size = args.batch_size * misc.get_world_size()
    
    print("Actual lr: {:.2e}".format(args.lr))
    print("Effective batch size: %d" % eff_batch_size)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # Set up optimizer with weight decay
    scaler = torch.amp.GradScaler()

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(0.9, 0.95), 
        weight_decay=args.weight_decay if hasattr(args, 'weight_decay') else cfg.WEIGHT_DECAY
    )
    
    print(optimizer)

    # Training loop
    best_metric = float('-inf')  # 用于保存最好指标，例如 SNR
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(scaler, model, data_loader_train, optimizer, device, epoch, log_writer=log_writer, args=args)
        
        # Perform online evaluation
        
        if args.fold != 'all':
            torch.cuda.empty_cache()
            metric = evaluate(model, data_loader_valid, device, epoch=epoch, log_writer=log_writer)
            
            current_metric = metric['snr']
            if current_metric > best_metric:
                best_metric = current_metric
                if misc.is_main_process():
                    print(f"Saving best model at epoch {epoch}, SNR={current_metric:.4f}")
                    misc.save_model(
                        args=args,
                        model_without_ddp=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        epoch_name="best"
                    )
        
        torch.cuda.empty_cache()

        # Save checkpoint periodically
        if epoch % args.save_last_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(
                args=args,
                model_without_ddp=model,
                optimizer=optimizer,
                epoch=epoch,
                epoch_name="last"
            )

        if misc.is_main_process() and log_writer is not None:
            log_writer.flush()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
