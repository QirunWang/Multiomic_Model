#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/03/26 13:30
# @Author  : Kieran Wang
# @File    : test4_dataloader.py
# @Software: PyCharm


import sys
sys.path.append("./")
from dataset.GetData import GetData_LLM,GetData_LLM_stat
from dataset.LoadData import MultiomicLoader
import torch
import numpy as np


MyDataset = GetData_LLM(pad_id=2,
                         pad_Pos_value=0,
                         pad_Expr_value=0,
                        gene_total_len=3072,
                        atac_total_len=6144,
                         gExpr_mask_frac=0.15,  # mask fraction of gExpr
                         aPos_mask_frac=0,  # mask fraction of aPos
                         aExpr_mask_frac=0.15,  # mask fraction of aExpr
                         gene_vocab_path="./references/gene_vocab_single_Vqiuping.json",
                         Database_dir = "/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_MultiomicsData_1M/clean/")
# MyDataset.get_all_h5ad_path()
temp1 = MyDataset.all_h5ad_path
len(MyDataset)

path,id=MyDataset.map_totalIDX_to_localIDX(10329)


RNA_data, ATAC_data = MyDataset.Get_h5ad_info(MyDataset.all_h5ad_path[path],id)

temp1= MyDataset[10329]


Mydataloader = MultiomicLoader(
                 dataset = MyDataset,
                 batch_size = 8,
                 shuffle=True
                 )

for batch in Mydataloader:
    RNA_module, ATAC_module = batch
    break


RNA_module['gID']
RNA_module['gPos']
RNA_module['gExpr']
RNA_module['gExpr_mask']

ATAC_module['aPos']
ATAC_module['aPos_mask']
ATAC_module['aExpr']
ATAC_module['aExpr_mask']

temp1 = torch.max(ATAC_module['aPos'],dim=1)
temp1.values/1e4



##stat
MyDataset = GetData_LLM_stat(pad_id=2,
                             pad_Pos_value=0,
                             pad_Expr_value=0,
                            gene_total_len=3072,
                            atac_total_len=6144,
                             gExpr_mask_frac=0.15,  # mask fraction of gExpr
                             aPos_mask_frac=0,  # mask fraction of aPos
                             aExpr_mask_frac=0.15,  # mask fraction of aExpr
                             gene_vocab_path="./references/gene_vocab_single_Vqiuping.json",
                             Database_dir = "/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_MultiomicsData_1M/clean2/")


temp1 = MyDataset.all_h5ad_path
len(MyDataset)

path,id=MyDataset.map_totalIDX_to_localIDX(10329)

temp0 = [x-1 for x in MyDataset.h5ad_cellnum_accumulate]
for i in temp0:
    RNA_data, ATAC_data = MyDataset[i]
    path, id = MyDataset.map_totalIDX_to_localIDX(i)
    RNA_data, ATAC_data = MyDataset.Get_h5ad_info(MyDataset.all_h5ad_path[path], id)
    if np.isnan(RNA_data['gene_stat']).any():
        print(f"{i} has NaN")

RNA_samples, ATAC_samples = MyDataset[10329]


Mydataloader = MultiomicLoader(
                 dataset = MyDataset,
                 batch_size = 8,
                 shuffle=True
                 )

for batch in Mydataloader:
    RNA_module, ATAC_module = batch
    break


RNA_module['gID']
RNA_module['gPos']
RNA_module['gExpr']
RNA_module['gExpr_mask']

ATAC_module['aPos']
ATAC_module['aPos_mask']
ATAC_module['aExpr']
ATAC_module['aExpr_mask']