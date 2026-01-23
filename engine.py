import torch
import util.misc as misc
import util.lr_sched as lr_sched
from util.metrics import snr_metric

def batch_to_device(batch, device):
    """
    Recursively move batch to device.
    Supports Tensor / dict / list / tuple.
    """
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)

    elif isinstance(batch, dict):
        return {
            k: batch_to_device(v, device)
            for k, v in batch.items()
        }

    elif isinstance(batch, (list, tuple)):
        return type(batch)(
            batch_to_device(v, device) for v in batch
        )

    else:
        # e.g. int, str, None
        return batch
    
    
def train_one_epoch(scaler, model, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        batch = batch_to_device(batch, device)
        
        x = batch["scan"]
        ecg_mv = batch['ecg_mv']
        
        # ===== forward =====
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x, ecg_mv, target_len=5120)
            loss = out['loss']

        if not torch.isfinite(loss):
            print(f"[Rank {misc.get_rank()}] Non-finite loss: {loss.item()}, skip step")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_value = loss.item()

        optimizer.zero_grad(set_to_none=True)

        # ===== backward（with scaler）=====
        scaler.scale(loss).backward()

        # ===== batch=1 unscale + grad clip =====
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # ===== optimizer step =====
        scaler.step(optimizer)
        scaler.update() 
        
        torch.cuda.synchronize()

        metric_logger.update(
            loss=loss_value,
            **{f"{k}": v for k, v in out.items() if k.startswith("metric_snr")}
        )
        
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)
    metric_logger.synchronize_between_processes()


@torch.no_grad()
def evaluate(model, data_loader, device, epoch=None, log_writer=None):
    """
    Args:
        model: torch.nn.Module
        data_loader: DataLoader
        device: torch.device
        epoch: current epoch
        log_writer: tensorboard writer
    Returns:
        metrics: dict, include avg_loss, avg_snr
    """
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Eval:'
    print_freq = 20

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        batch = batch_to_device(batch, device)
        
        x = batch["scan"]
        labels = batch["label"]
        ecg_mv = batch["ecg_mv"]
        
        target_len = 5120

        with torch.amp.autocast('cuda', dtype=torch.float16):
            out = model(x, labels, ecg_mv=None, target_len=target_len)
            
        snr_mean = snr_metric(out[f'y_mv_{target_len}'], ecg_mv[f'{target_len}'])
        metric_logger.update(snr=snr_mean)

    # 汇总多卡结果
    snr_value_reduce = misc.all_reduce_mean(metric_logger.meters['snr'].global_avg)
    print(f" * [Epoch {epoch}] Avg SNR {snr_value_reduce:.4f}")

    if log_writer is not None and epoch is not None:
        log_writer.add_scalar('valid_snr', snr_value_reduce, epoch)

    return {
        'snr': snr_value_reduce,
    }
