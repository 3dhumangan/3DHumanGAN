import torch
from .preprocessor import get_preprocessor
from .datasets import SHHQDataset

def get_dataset(class_name, subsample=None, proc_batch_size=1, num_workers=1, **kwargs):
    dataset = globals()[class_name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=proc_batch_size,
        shuffle=kwargs.get("shuffle", True),
        drop_last=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    return dataloader, 3


def get_dataset_distributed(class_name, world_size, rank, proc_batch_size, num_workers=4, **kwargs):
    dataset = globals()[class_name](**kwargs)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=kwargs.get("shuffle", True)
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=proc_batch_size,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
    )

    return dataloader, 3


