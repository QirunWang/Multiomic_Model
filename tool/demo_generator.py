#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/03/19 09:42
# @Author  : Kieran Wang
# @File    : demo_generator.py
# @Software: PyCharm


import numpy as np
import torch

class demo_input:
    def __init__(self,
                 n_batch, # number of elements in the minibatch

                 gene_len,  # length for gene-related tensor (gID, gPos, gExpr, gExpr_mask)
                 atac_len,  # length for atac-related tensor (aPos, aPos_mask, aExpr, aExpr_mask)
                 gene_total_len, # length for gene-related tensor with pading or crop
                 atac_total_len,  # length for atac-related tensor with pading or crop

                 gExpr_mask_frac = 0.15,  # mask fraction of gExpr
                 aPos_mask_frac = 0,  # mask fraction of aPos
                 aExpr_mask_frac = 0.15,  #mask fraction of aExpr

                 max_gene_num=1000, # gene ID range
                 max_expr=10,       # RNA and ATAC log1p expr range
                 max_pos=1e6):      # ATAC postion range
        """
        Initializes the generator with:
        :param n_batch: number of elements in the minibatch
        :param gene_len: length for gene-related lists (gID, gExpr, gPos)
        :param atac_len: length for atac-related lists (aPos, aExpr)
        """
        self.n_batch = n_batch
        self.gene_len = gene_len
        self.atac_len = atac_len
        self.gene_total_len = gene_total_len
        self.atac_total_len = atac_total_len

        self.gExpr_mask_frac = gExpr_mask_frac
        self.aPos_mask_frac = aPos_mask_frac
        self.aExpr_mask_frac = aExpr_mask_frac

        self.max_gene_num = max_gene_num
        self.max_expr = max_expr
        self.max_pos = int(max_pos)

    def _pad_or_crop(self, tensor, total_len):
        """Pad with zeros if tensor length is less than total_len, or crop if longer."""
        current_len = tensor.size(0)
        if current_len < total_len:
            pad_tensor = torch.zeros(total_len - current_len, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad_tensor], dim=0)
        else:
            tensor = tensor[:total_len]
        return tensor

    def generate_data(self):
        batch_gID = []
        batch_gPos = []
        batch_gExpr = []
        batch_gExpr_mask = []

        batch_aPos = []
        batch_aPos_mask = []
        batch_aExpr = []
        batch_aExpr_mask = []

        for _ in range(self.n_batch):
            # Gene-related tensors
            # gID: unique random integers from 0 to max_gene_num-1.
            if self.gene_len > self.max_gene_num:
                raise ValueError("gene_len cannot be greater than max_gene_num for unique values.")
            gID = torch.randperm(self.max_gene_num)[:self.gene_len]
            gID = self._pad_or_crop(gID, self.gene_total_len)

            # gPos: unique random integers between 1 and max_pos (inclusive)
            if self.gene_len > self.max_pos:
                raise ValueError("gene_len cannot be greater than max_pos for unique gene positions.")
            gPos = torch.randperm(self.max_pos)[:self.gene_len] + 1
            gPos = self._pad_or_crop(gPos, self.gene_total_len)

            # gExpr: random floats between 0 and max_expr
            gExpr = torch.rand(self.gene_len) * self.max_expr
            gExpr = self._pad_or_crop(gExpr, self.gene_total_len)

            # gExpr_mask: binary mask with probability gExpr_mask_frac for 1
            gExpr_mask = (torch.rand(self.gene_len) < self.gExpr_mask_frac).to(torch.int64)
            gExpr_mask = self._pad_or_crop(gExpr_mask, self.gene_total_len)

            # ATAC-related tensors
            # aPos: unique random integers between 1 and max_pos (inclusive)
            if self.atac_len > self.max_pos:
                raise ValueError("atac_len cannot be greater than max_pos for unique atac positions.")
            aPos = torch.randperm(self.max_pos)[:self.atac_len] + 1
            aPos = self._pad_or_crop(aPos, self.atac_total_len)

            # aPos_mask: binary mask with probability aPos_mask_frac for 1
            aPos_mask = (torch.rand(self.atac_len) < self.aPos_mask_frac).to(torch.int64)
            aPos_mask = self._pad_or_crop(aPos_mask, self.atac_total_len)

            # aExpr: random floats between 0 and max_expr
            aExpr = torch.rand(self.atac_len) * self.max_expr
            aExpr = self._pad_or_crop(aExpr, self.atac_total_len)

            # aExpr_mask: binary mask with probability aExpr_mask_frac for 1
            aExpr_mask = (torch.rand(self.atac_len) < self.aExpr_mask_frac).to(torch.int64)
            aExpr_mask = self._pad_or_crop(aExpr_mask, self.atac_total_len)

            batch_gID.append(gID)
            batch_gPos.append(gPos)
            batch_gExpr.append(gExpr)
            batch_gExpr_mask.append(gExpr_mask)

            batch_aPos.append(aPos)
            batch_aPos_mask.append(aPos_mask)
            batch_aExpr.append(aExpr)
            batch_aExpr_mask.append(aExpr_mask)

        # Stack the tensors to form the final batch with shapes:
        # gene tensors: [n_batch, gene_total_len]
        # atac tensors: [n_batch, atac_total_len]
        batch_gID = torch.stack(batch_gID)
        batch_gPos = torch.stack(batch_gPos)
        batch_gExpr = torch.stack(batch_gExpr)
        batch_gExpr_mask = torch.stack(batch_gExpr_mask)

        batch_aPos = torch.stack(batch_aPos)
        batch_aPos_mask = torch.stack(batch_aPos_mask)
        batch_aExpr = torch.stack(batch_aExpr)
        batch_aExpr_mask = torch.stack(batch_aExpr_mask)

        RNA_module = {
            "gID": batch_gID,
            "gPos": batch_gPos,
            "gExpr": batch_gExpr,
            "gExpr_mask": batch_gExpr_mask,}

        ATAC_module = {
            "aPos": batch_aPos,
            "aPos_mask": batch_aPos_mask,
            "aExpr": batch_aExpr,
            "aExpr_mask": batch_aExpr_mask
        }
        return RNA_module, ATAC_module

# Example usage:
if __name__ == "__main__":
    generator = demo_input(
        n_batch=3,
        gene_len=5,
        atac_len=10,
        gene_total_len=10,
        atac_total_len=15,
        gExpr_mask_frac=0.15,
        aPos_mask_frac=0,
        aExpr_mask_frac=0.15,
        max_gene_num=1000,
        max_expr=10,
        max_pos=1000000  # Using an integer value
    )
    RNA_module, ATAC_module = generator.generate_data()

    RNA_module['gID']
    RNA_module['gPos']
    RNA_module['gExpr']
    RNA_module['gExpr_mask']

    ATAC_module['aPos']
    ATAC_module['aPos_mask']
    ATAC_module['aExpr']
    ATAC_module['aExpr_mask']











