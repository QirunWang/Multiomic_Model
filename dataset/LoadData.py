#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/03/27 16:55
# @Author  : Kieran Wang
# @File    : DataLoader.py
# @Software: PyCharm



import random
from torch.utils.data import Sampler,DistributedSampler, DataLoader
import torch.distributed as dist
import torch
import numpy as np
import math

class DistributedIndicesSampler(DistributedSampler):
    def __init__(self,
                 dataset,
                 drawable_indices, # a list containing drawable indices
                 shuffle=True,
                 ):
        super().__init__(dataset=dataset,shuffle=shuffle)
        self.drawable_indices = drawable_indices

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(self.drawable_indices)
            if padding_size <= len(self.drawable_indices):
                self.drawable_indices += self.drawable_indices[:padding_size]
            else:
                self.drawable_indices += (self.drawable_indices * math.ceil(padding_size / len(self.drawable_indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            self.drawable_indices = self.drawable_indices[:self.total_size]
        assert len(self.drawable_indices) == self.total_size

    def __iter__(self):
        # subsample
        indices = self.drawable_indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler and shuffle the drawable index
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            random_order = torch.randperm(len(self.drawable_indices), generator=g).tolist()  # type: ignore[arg-type]
            self.drawable_indices = [self.drawable_indices[x] for x in random_order]


class IndicesSampler(Sampler):
    def __init__(self, dataset, drawable_indices, shuffle=True):
        """
        Args:
            dataset: The dataset (this argument is kept for compatibility with DataLoader).
            drawable_indices (list): A list of indices specifying which samples to draw.
            shuffle (bool): Whether to shuffle the indices at the start of each iteration.
        """
        self.dataset = dataset
        # Create a local copy of the drawable indices
        self.drawable_indices = list(drawable_indices)
        self.shuffle = shuffle

    def __iter__(self):
        # Optionally shuffle the indices for each new epoch.
        indices = self.drawable_indices.copy()
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        # Total number of indices available for sampling.
        return len(self.drawable_indices)

class MultiomicLoader(DataLoader):
    def __init__(self,
                 dataset,
                 batch_size,
                 drawable_indices, # a list containing drawable indices
                 num_workers=8,
                 shuffle=True,
                 distributed=False):
        self.drawable_indices = drawable_indices
        self.distributed = distributed
        # Create a DistributedSampler if using DDP
        if self.distributed:
            self.sampler = DistributedIndicesSampler(dataset, self.drawable_indices, shuffle=shuffle)
            shuffle = False  # Disable default shuffling; sampler handles it.
        else:
            self.sampler = IndicesSampler(dataset, self.drawable_indices, shuffle=shuffle)

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            sampler=self.sampler,
            collate_fn=self.collate_fn
        )

        if self.distributed:
            self.num_batches = len(self.drawable_indices) // (self.batch_size * dist.get_world_size())
        else:
            self.num_batches = len(self.drawable_indices) // self.batch_size

    def set_epoch(self, epoch):
        # Ensures different shuffling at each epoch
        if self.distributed and self.sampler is not None:
            self.sampler.set_epoch(epoch)


    def collate_fn(self, batch):
        # Unzip the list of tuples into two lists:
        try:
            RNA_samples, ATAC_samples = zip(*batch)
        except ValueError as e:
            raise ValueError("Each item in the batch should be a tuple (RNA_sc, ATAC_sc)") from e

        # For RNA, stack each field using numpy.array to speed up tensor creation.
        RNA_module = {
            "gID": torch.from_numpy(np.array([sample["gID"] for sample in RNA_samples])).int(),
            "gPos": torch.from_numpy(np.array([sample["gPos"] for sample in RNA_samples])).float(),
            "gExpr": torch.from_numpy(np.array([sample["gExpr"] for sample in RNA_samples])).float(),
            "gExpr_mask": torch.from_numpy(np.array([sample["gExpr_mask"] for sample in RNA_samples])).int()
        }

        # For ATAC, stack each field similarly.
        ATAC_module = {
            "aPos": torch.from_numpy(np.array([sample["aPos"] for sample in ATAC_samples])).float(),
            "aPos_mask": torch.from_numpy(np.array([sample["aPos_mask"] for sample in ATAC_samples])).int(),
            "aExpr": torch.from_numpy(np.array([sample["aExpr"] for sample in ATAC_samples])).float(),
            "aExpr_mask": torch.from_numpy(np.array([sample["aExpr_mask"] for sample in ATAC_samples])).int()
        }

        return RNA_module, ATAC_module



class MultiomicLoader_stat(MultiomicLoader):
    def __init__(self,
                 dataset,
                 batch_size,
                 drawable_indices, # a list containing drawable indices
                 num_workers=8,
                 shuffle=True,
                 distributed=False):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            drawable_indices=drawable_indices,
            num_workers=num_workers,
            shuffle=shuffle,
            distributed = distributed
        )

    def collate_fn(self, batch):
        # Unzip the list of tuples into two lists:
        try:
            RNA_samples, ATAC_samples = zip(*batch)
        except ValueError as e:
            raise ValueError("Each item in the batch should be a tuple (RNA_sc, ATAC_sc)") from e

        # For RNA, stack each field using numpy.array to speed up tensor creation.
        RNA_module = {
            "gID": torch.from_numpy(np.array([sample["gID"] for sample in RNA_samples])).int(),
            "gPos": torch.from_numpy(np.array([sample["gPos"] for sample in RNA_samples])).float(),
            "gExpr": torch.from_numpy(np.array([sample["gExpr"] for sample in RNA_samples])).float(),
            "gExpr_mask": torch.from_numpy(np.array([sample["gExpr_mask"] for sample in RNA_samples])).int(),
            "gStat": torch.from_numpy(np.array([sample["gStat"] for sample in RNA_samples])).float(),
        }

        # For ATAC, stack each field similarly.
        ATAC_module = {
            "aPos": torch.from_numpy(np.array([sample["aPos"] for sample in ATAC_samples])).float(),
            "aPos_mask": torch.from_numpy(np.array([sample["aPos_mask"] for sample in ATAC_samples])).int(),
            "aExpr": torch.from_numpy(np.array([sample["aExpr"] for sample in ATAC_samples])).float(),
            "aExpr_mask": torch.from_numpy(np.array([sample["aExpr_mask"] for sample in ATAC_samples])).int()
        }

        return RNA_module, ATAC_module




# test
if __name__ == "__main__":
    import sys
    sys.path.append("./")
    from dataset.GetData import GetData_LLM, GetData_LLM_stat

    # MyDataset = GetData_LLM(pad_id=2,
    #                         pad_Pos_value=0,
    #                         pad_Expr_value=0,
    #                         gene_total_len=3072,
    #                         atac_total_len=6144,
    #                         gExpr_mask_frac=0.15,  # mask fraction of gExpr
    #                         aPos_mask_frac=0,  # mask fraction of aPos
    #                         aExpr_mask_frac=0.15,  # mask fraction of aExpr
    #                         gene_vocab_path="./references/gene_vocab_single_Vqiuping.json",
    #                         Database_dir="/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_MultiomicsData_1M/clean/")
    # Mydataloader = MultiomicLoader(
    #     dataset=MyDataset,
    #     batch_size=8,
    #     num_workers=0,
    #     shuffle=True
    # )

    MyDataset = GetData_LLM_stat(pad_id=2,
                                 pad_Pos_value=0,
                                 pad_Expr_value=0,
                                 gene_total_len=3072,
                                 atac_total_len=6144,
                                 gExpr_mask_frac=0.15,  # mask fraction of gExpr
                                 aPos_mask_frac=0,  # mask fraction of aPos
                                 aExpr_mask_frac=0.15,  # mask fraction of aExpr
                                 pos_scaling_factor=1e6,
                                 gene_vocab_path="./references/gene_vocab_single_Vqiuping.json",
                                 Database_dir="/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_MultiomicsData_1M/clean2/")
    drawable_indices = list(np.load("./references/drawableidx_train_20250403.npy"))
    Mydataloader = MultiomicLoader_stat(
        dataset=MyDataset,
        drawable_indices=drawable_indices,
        batch_size=8,
        num_workers=0,
        shuffle=True
    )


    for batch in Mydataloader:
        RNA_module, ATAC_module = batch
        break

    RNA_module['gID']
    RNA_module['gPos']
    RNA_module['gExpr']
    RNA_module['gExpr_mask']
    RNA_module['gStat']

    ATAC_module['aPos']
    ATAC_module['aPos_mask']
    ATAC_module['aExpr']
    ATAC_module['aExpr_mask']




