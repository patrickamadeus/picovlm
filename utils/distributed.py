import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def is_master():
    return get_rank() == 0


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda", get_local_rank())
    return torch.device("cpu")


def init_distributed(timeout_minutes=30):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1 or is_distributed():
        return False
    rank = int(os.environ.get("RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_method = os.environ.get("DIST_INIT_METHOD")
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=timeout_minutes),
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())
    return True


def destroy_distributed():
    if is_distributed():
        dist.destroy_process_group()


def barrier():
    if is_distributed():
        dist.barrier()


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_model(model):
    if not is_distributed():
        return model
    if torch.cuda.is_available():
        local_rank = get_local_rank()
        return DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    return DistributedDataParallel(model)


def sum_scalar(value, device):
    if not is_distributed():
        return value
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to(device=device, dtype=torch.float64)
    else:
        tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def gather_objects(obj):
    if not is_distributed():
        return [obj]
    gathered = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered
