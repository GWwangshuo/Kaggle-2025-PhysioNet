import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist

class SmoothedValue:
    def __init__(self, window_size=20, fmt="{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], device="cuda")
        dist.all_reduce(t)
        self.count = int(t[0].item())
        self.total = t[1].item()

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque)).mean().item()

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
        )


class MetricLogger:
    """
    Metric logger with ETA, data time, iteration time.
    DDP-safe, multi-node friendly.
    """

    def __init__(self, delimiter="\t", print_on_rank0=True):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.print_on_rank0 = print_on_rank0
        
    def add_meter(self, name, meter): 
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def __str__(self):
        return self.delimiter.join(
            f"{name}: {meter}" for name, meter in self.meters.items()
        )

    def log_every(self, iterable, print_freq, header=""):
        """
        Args:
            iterable: dataloader or iterable
            print_freq: log frequency
            header: string header
        """
        i = 0
        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        num_iters = len(iterable)
        space_fmt = ":" + str(len(str(num_iters))) + "d"

        log_msg = [
            header,
            f"[{{0{space_fmt}}}/{{1}}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]

        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")

        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        is_master = rank == 0 or not self.print_on_rank0

        for obj in iterable:
            # data loading time
            data_time.update(time.time() - end)

            yield obj

            # iteration time
            iter_time.update(time.time() - end)

            if is_master and (i % print_freq == 0 or i == num_iters - 1):
                eta_seconds = iter_time.global_avg * (num_iters - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            num_iters,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            num_iters,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))

        if is_master:
            print(
                f"{header} Total time: {total_time_str} "
                f"({total_time / max(1, num_iters):.4f} s / it)"
            )


def setup_for_distributed(is_master):
    """
    Disable printing on non-master processes
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print(f"[{now}]", *args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # torchrun
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
        args.dist_url = "env://"

    # SLURM
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.world_size = int(os.environ["SLURM_NTASKS"])
        args.gpu = int(os.environ["SLURM_LOCALID"])
        args.dist_url = "env://"

    else:
        print("Not using distributed mode")
        args.distributed = False
        setup_for_distributed(True)
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"

    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}",
        flush=True,
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def add_weight_decay(model, weight_decay=0, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list or 'diffloss' in name:
            no_decay.append(param)  # no weight decay on bias, norm and diffloss
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def all_reduce_mean(x):
    if get_world_size() > 1:
        t = torch.tensor(x, device="cuda")
        dist.all_reduce(t)
        t /= get_world_size()
        return t.item()
    return x


def save_model(args, model_without_ddp, optimizer, epoch, epoch_name="last"):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_on_master(
        {
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": args,
        },
        output_dir / f"checkpoint-{epoch_name}.pth",
    )