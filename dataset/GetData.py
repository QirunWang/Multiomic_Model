#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/03/26 10:26
# @Author  : Kieran Wang
# @File    : GetData.py
# @Software: PyCharm

import time

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import lmdb
import json
import os
import random
from joblib import Parallel, delayed
import torch
import torch.distributed as dist

import h5py
from scipy.sparse import csr_matrix


#file_path="./data/subdir_1/test_scdata.h5ad"
class Basic_Dataset(Dataset):
    def __init__(self,
                 keep_zero_gene = False, # whether to keep 0 expr genes

                 pad_id = 2,
                 pad_Pos_value=0,
                 pad_Expr_value=0,

                 gene_total_len = 1024,  # length for gene-related tensor (gID, gPos, gExpr, gExpr_mask) with pading or crop
                 atac_total_len = 8192,  # length for atac-related tensor (aPos, aPos_mask, aExpr, aExpr_mask) with pading or crop

                 gExpr_mask_frac=0.15,  # mask fraction of gExpr
                 aPos_mask_frac=0,  # mask fraction of aPos
                 aExpr_mask_frac=0.15,  # mask fraction of aExpr

                 gene_vocab_path = "./references/gene_vocab_single_Vqiuping.json",
                 ):
        os.chdir('/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2')

        self.keep_zero_gene = keep_zero_gene
        self.pad_id = pad_id
        self.pad_Pos_value = pad_Pos_value
        self.pad_Expr_value = pad_Expr_value
        self.gene_total_len = gene_total_len
        self.atac_total_len = atac_total_len
        self.gExpr_mask_frac = gExpr_mask_frac
        self.aPos_mask_frac = aPos_mask_frac
        self.aExpr_mask_frac = aExpr_mask_frac

        with open(gene_vocab_path, "r") as f:
            self.gene_Symbol2id=json.load(f)  # Convert JSON to dictionary

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class GetData_LLM(Basic_Dataset):
    def __init__(self,
                 pad_id=2,
                 pad_Pos_value=0,
                 pad_Expr_value=0,

                 gene_total_len=3072,
                 # length for gene-related tensor (gID, gPos, gExpr, gExpr_mask) with pading or crop
                 atac_total_len=6144,
                 # length for atac-related tensor (aPos, aPos_mask, aExpr, aExpr_mask) with pading or crop

                 gExpr_mask_frac=0.15,  # mask fraction of gExpr
                 aPos_mask_frac=0,  # mask fraction of aPos
                 aExpr_mask_frac=0.15,  # mask fraction of aExpr

                 gene_vocab_path="./references/gene_vocab_single_Vqiuping.json",

                 Database_dir = "./demo_data/cleaned"):
        super().__init__(
                         pad_id=pad_id,
                         pad_Pos_value=pad_Pos_value,
                         pad_Expr_value=pad_Expr_value,
                         gene_total_len=gene_total_len,
                         atac_total_len=atac_total_len,
                         gExpr_mask_frac=gExpr_mask_frac,
                         aPos_mask_frac=aPos_mask_frac,
                         aExpr_mask_frac=aExpr_mask_frac,
                         gene_vocab_path=gene_vocab_path
                        )

        self.Database_dir = Database_dir
        print(f"h5ad database dir: {self.Database_dir}")

        self.get_all_h5ad_path()
        print(f"h5ad file number: {len(self.all_h5ad_path)}")
        print(f"h5ad file names: {self.all_h5ad_path}")

        self.count_all_h5ad_cell()
        print(f"Cell number for every h5ad: {self.h5ad_cellnum}")
        print(f"Cell number accumulation: {self.h5ad_cellnum_accumulate}")

        print(f"Total cell number: {self.__len__()}")


    def __getitem__(self, total_idx):
        # get the h5ad and the local idx of the total_idx
        h5ad_idx, local_idx = self.map_totalIDX_to_localIDX(total_idx)
        h5ad_path = self.all_h5ad_path[h5ad_idx]

        # retrieve data
        RNA_data, ATAC_data = self.Get_h5ad_info(h5ad_path,local_idx)

        # map gene symbol back to vocab ID
        temp1 = RNA_data['gene_symbols']
        mapped_ids = np.array([self.gene_Symbol2id.get(symbol, None) for symbol in temp1])

        # delete unmaped genes (None) in all RNA data
        gID = np.array([mapped_ids[i] for i in range(len(mapped_ids)) if mapped_ids[i] is not None])
        gPos = np.array([RNA_data['gene_positions'][i] for i in range(len(mapped_ids)) if mapped_ids[i] is not None])
        gExpr = np.array([RNA_data['gene_values'][i] for i in range(len(mapped_ids)) if mapped_ids[i] is not None])

        # set mask to gExpr
        gExpr_mask = (np.random.rand(len(gExpr)) < self.gExpr_mask_frac).astype(int)

        # padding and cropping RNA data
        current_length = gID.shape[0]
        if current_length < self.gene_total_len:
            # Calculate how many zeros to pad
            pad_length = self.gene_total_len - current_length
            # padding gID
            padding = np.full(pad_length, self.pad_id, dtype=gID.dtype)
            gID = np.concatenate([gID, padding])
            # padding gPos
            padding = np.full(pad_length, self.pad_Pos_value, dtype=gPos.dtype)
            gPos = np.concatenate([gPos, padding])
            # padding gExpr
            padding = np.full(pad_length, self.pad_Expr_value,dtype=gExpr.dtype)
            gExpr = np.concatenate([gExpr, padding])
            # padding gExpr_mask
            padding = np.full(pad_length, 0, dtype=gExpr_mask.dtype)
            gExpr_mask = np.concatenate([gExpr_mask, padding])
        else:
            # Crop the array to random n elements
            indices = np.random.choice(current_length, self.gene_total_len, replace=False)
            indices = np.sort(indices)
            # cropping gID
            gID = gID[indices]
            # cropping gPos
            gPos = gPos[indices]
            # cropping gExpr
            gExpr = gExpr[indices]
            # cropping gExpr_mask
            gExpr_mask = gExpr_mask[indices]


        # get aPos and aExpr
        aPos = ATAC_data['peak_positions']
        aExpr = ATAC_data['peak_values']

        # set mask to aPos
        aPos_mask = (np.random.rand(len(aPos)) < self.aPos_mask_frac).astype(int)
        aExpr_mask = (np.random.rand(len(aExpr)) < self.aExpr_mask_frac).astype(int)

        # padding and cropping ATAC data
        current_length = aPos.shape[0]
        if current_length < self.atac_total_len:
            # Calculate how many zeros to pad
            pad_length = self.atac_total_len - current_length
            # padding aPos
            padding = np.full(pad_length, self.pad_Pos_value, dtype=aPos.dtype)
            aPos = np.concatenate([aPos, padding])
            # padding aPos_mask
            padding = np.full(pad_length, 0, dtype=aPos_mask.dtype)
            aPos_mask = np.concatenate([aPos_mask, padding])
            # padding aExpr
            padding = np.full(pad_length, self.pad_Expr_value, dtype=aExpr.dtype)
            aExpr = np.concatenate([aExpr, padding])
            # padding aExpr_mask
            padding = np.full(pad_length, 0, dtype=aExpr_mask.dtype)
            aExpr_mask = np.concatenate([aExpr_mask, padding])
        else:
            # Crop the array to random n elements
            indices = np.random.choice(current_length, self.atac_total_len, replace=False)
            indices = np.sort(indices)
            # cropping aPos
            aPos = aPos[indices]
            # cropping aPos_mask
            aPos_mask = aPos_mask[indices]
            # cropping aExpr
            aExpr = aExpr[indices]
            # cropping aExpr_mask
            aExpr_mask = aExpr_mask[indices]

        # group data
        RNA_sc = {
            "gID": gID,
            "gPos": gPos,
            "gExpr": gExpr,
            "gExpr_mask": gExpr_mask,
        }
        ATAC_sc = {
            "aPos": aPos,
            "aPos_mask": aPos_mask,
            "aExpr": aExpr,
            "aExpr_mask": aExpr_mask
        }
        return RNA_sc, ATAC_sc


    def __len__(self):
        return sum(self.h5ad_cellnum)


    def get_all_h5ad_path(self):
        """
        get all the path of all h5ad in the database dir
        """
        data_sets = os.listdir(self.Database_dir)
        i=1
        All_path = []
        for i in range(len(data_sets)):
            temp1 = os.path.join(self.Database_dir,data_sets[i])
            data_file = os.listdir(temp1)
            data_path = [os.path.join(temp1,x) for x in data_file]
            All_path = All_path + data_path

        self.all_h5ad_path = All_path


    def count_all_h5ad_cell(self):
        """
        get info about the database including:
            1. cell number
            2. What is the index for cell in a file
        """
        self.h5ad_cellnum = np.array([self.Count_h5ad_cellnum(x) for x in self.all_h5ad_path])
        self.h5ad_cellnum_accumulate = [sum(self.h5ad_cellnum[0:(i+1)]) for i in range(len(self.h5ad_cellnum))]



    def map_totalIDX_to_localIDX(self,totalIDX):
        """
        Given an index among all cells in the database,
        retrieve which h5ad the cell is from, as well as the index of the cell in that h5ad.
        """
        for i, chunk_end in enumerate(self.h5ad_cellnum_accumulate):
            if totalIDX < chunk_end:
                chunk_start = self.h5ad_cellnum_accumulate[i - 1] if i > 0 else 0
                localIDX = totalIDX - chunk_start
                h5ad_index = i
                return h5ad_index, localIDX
        raise IndexError("Index out of range.")

    def input_stat(self, RNA_minibatch, ATAC_minibatch):
        """
        This function is used to get the statistic aspect of a minibatch for monitoring purpose
        For minibatch, returns genes' and peaks' length mean, std, max and min
        """
        RNA_lengths = (RNA_minibatch['gID'] != self.pad_id).sum(dim=1)
        ATAC_lengths = (ATAC_minibatch['aPos'] != 0).sum(dim=1)

        # Calculate statistics: mean, min, max, and standard deviation.
        res_stat = {
            'RNA_len_min': RNA_lengths.min(),
            'RNA_len_max': RNA_lengths.max(),
            'RNA_len_avg': RNA_lengths.float().mean(),
            'RNA_len_std': RNA_lengths.float().std(),
            'ATAC_len_min': ATAC_lengths.min(),
            'ATAC_len_max': ATAC_lengths.max(),
            'ATAC_len_avg': ATAC_lengths.float().mean(),
            'ATAC_len_std': ATAC_lengths.float().std()
        }

        return res_stat

    @staticmethod
    def Count_h5ad_cellnum(file_path):
        with h5py.File(file_path, 'r') as f:
            num_cells = f['X']['indptr'][:].shape[0] - 1
        return num_cells


    @staticmethod
    def Get_h5ad_structure(file_path):
        def print_structure(name, obj):
            indent = '  ' * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name} (Dataset) - shape: {obj.shape}, dtype: {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name} (Group)")
            else:
                print(f"{indent}{name} (Unknown)")

        with h5py.File(file_path, 'r') as f:
            f.visititems(print_structure)


    @staticmethod
    def Get_h5ad_info(file_path,cell_index):
        """
        This function is used to extract the expression data of the cell with cell_index in that file_path
        """
        with h5py.File(file_path, 'r') as f:
            # Access the CSR components from the "X" group
            X_group = f['X']
            data = X_group['data']
            indices = X_group['indices']
            indptr = X_group['indptr']

            # Get the nonzero entries for the specified cell
            # a csr matrix is concatenated into a long list by its rows without 0.
            # the indptr is a n_cell+1 list, where the value under index i and i+1 indicates a cell's start idx and end idx
            # for example, if i and i+1 elements in indptr is 100 and 200, respecitvely,
            # then it means the ith cell's data is from data list[100] to data list[199]
            start = indptr[cell_index]
            end = indptr[cell_index + 1]
            cell_data = data[start:end]
            cell_feature_indices = indices[start:end]

            # Read feature annotations from the "var" group
            var_group = f['var']

            # Feature names are stored in var/_index
            feature_names = var_group['_index'][:]
            feature_names = np.array([
                name.decode('utf-8') if isinstance(name, bytes) else str(name)
                for name in feature_names
            ])

            # Extract the "position" information from var (assuming it's stored as integers)
            positions = var_group['position'][:]  # shape: (num_features,)

            # Get feature type information from var/feature_types
            ft_group = var_group['feature_types']
            codes = ft_group['codes'][:]  # shape: (num_features,)
            categories = ft_group['categories'][:]
            categories = np.array([
                cat.decode('utf-8') if isinstance(cat, bytes) else str(cat)
                for cat in categories
            ])

            # Determine the code for gene expression and peaks.
            gene_code = None
            peak_code = None
            for idx, cat in enumerate(categories):
                if cat.lower() in ["gene expression", "gene", "gene_expr"]:
                    gene_code = idx
                elif cat.lower() in ["peaks", "peak"]:
                    peak_code = idx
            if gene_code is None or peak_code is None:
                raise ValueError("Could not identify gene and peak feature codes based on categories.")

            # For the features present in this cell, determine which are genes and which are peaks.
            cell_codes = codes[cell_feature_indices]
            gene_mask = (cell_codes == gene_code)
            peak_mask = (cell_codes == peak_code)

            gene_feature_indices = cell_feature_indices[gene_mask]
            gene_values = cell_data[gene_mask]
            peak_feature_indices = cell_feature_indices[peak_mask]
            peak_values = cell_data[peak_mask]

            # Map feature indices to feature names and positions
            gene_symbols = feature_names[gene_feature_indices]
            gene_positions = positions[gene_feature_indices]
            peak_symbols = feature_names[peak_feature_indices]
            peak_positions = positions[peak_feature_indices]

            RNA_data = {'gene_symbols': gene_symbols,
                        'gene_positions':gene_positions,
                        'gene_values':gene_values
                        }
            ATAC_data = {'peak_symbols': peak_symbols,
                         'peak_positions':peak_positions,
                         'peak_values':peak_values
                         }

            return RNA_data, ATAC_data

    def exprot_drawableidx(self, export_path = "./references/",file_prefix = None, val_frac=0.1):
        # split into test and val
        self.val_frac = val_frac
        val_size = int(sum(self.h5ad_cellnum) * self.val_frac)

        val_idx = torch.randperm(sum(self.h5ad_cellnum))[:val_size].numpy()
        train_idx = torch.randperm(sum(self.h5ad_cellnum))[val_size:].numpy()

        if file_prefix is not None:
            np.save(export_path + file_prefix + '_drawableidx_valid.npy', val_idx)
            np.save(export_path + file_prefix + '_drawableidx_train.npy', train_idx)
        else:
            np.save(export_path + 'drawableidx_valid.npy', val_idx)
            np.save(export_path + 'drawableidx_train.npy', train_idx)




class GetData_LLM_stat(GetData_LLM):
    """ This class add stat info for RNA in each samples"""
    def __init__(self,
                 pad_id=2,
                 pad_Pos_value=0,
                 pad_Expr_value=0,

                 gene_total_len=3072,
                 # length for gene-related tensor (gID, gPos, gExpr, gExpr_mask) with pading or crop
                 atac_total_len=6144,
                 # length for atac-related tensor (aPos, aPos_mask, aExpr, aExpr_mask) with pading or crop

                 gExpr_mask_frac=0.15,  # mask fraction of gExpr
                 aPos_mask_frac=0,  # mask fraction of aPos
                 aExpr_mask_frac=0.15,  # mask fraction of aExpr

                 gstat=False, # whether to get statistic infomation into samples

                 gene_vocab_path="./references/gene_vocab_single_Vqiuping.json",

                 Database_dir = "./demo_data/cleaned"):
        super().__init__(
                         pad_id=pad_id,
                         pad_Pos_value=pad_Pos_value,
                         pad_Expr_value=pad_Expr_value,
                         gene_total_len=gene_total_len,
                         atac_total_len=atac_total_len,
                         gExpr_mask_frac=gExpr_mask_frac,
                         aPos_mask_frac=aPos_mask_frac,
                         aExpr_mask_frac=aExpr_mask_frac,
                         gene_vocab_path=gene_vocab_path,
                         Database_dir=Database_dir
                        )
        self.gstat = gstat

    def __getitem__(self, total_idx):
        # get the h5ad and the local idx of the total_idx
        h5ad_idx, local_idx = self.map_totalIDX_to_localIDX(total_idx)
        h5ad_path = self.all_h5ad_path[h5ad_idx]

        # retrieve data
        RNA_data, ATAC_data = self.Get_h5ad_info(h5ad_path,local_idx)

        # map gene symbol back to vocab ID
        temp1 = RNA_data['gene_symbols']
        mapped_ids = np.array([self.gene_Symbol2id.get(symbol, None) for symbol in temp1])

        # delete unmaped genes (None) in all RNA data
        gID = np.array([mapped_ids[i] for i in range(len(mapped_ids)) if mapped_ids[i] is not None])
        gPos = np.array([RNA_data['gene_positions'][i] for i in range(len(mapped_ids)) if mapped_ids[i] is not None])
        gExpr = np.array([RNA_data['gene_values'][i] for i in range(len(mapped_ids)) if mapped_ids[i] is not None])

        valid_mask = np.array([mapped_id is not None for mapped_id in mapped_ids])
        gStat = RNA_data['gene_stat'][valid_mask, :]

        # random order the genes
        random_indices = np.random.permutation(len(gID))
        gID = gID[random_indices]
        gPos = gPos[random_indices]
        gExpr = gExpr[random_indices]
        gStat = gStat[random_indices,:]

        # set mask to gExpr
        gExpr_mask = (np.random.rand(len(gExpr)) < self.gExpr_mask_frac).astype(int)

        # padding and cropping RNA data
        current_length = gID.shape[0]
        if current_length < self.gene_total_len:
            # Calculate how many zeros to pad
            pad_length = self.gene_total_len - current_length
            # padding gID
            padding = np.full(pad_length, self.pad_id, dtype=gID.dtype)
            gID = np.concatenate([gID, padding])
            # padding gPos
            padding = np.full(pad_length, self.pad_Pos_value, dtype=gPos.dtype)
            gPos = np.concatenate([gPos, padding])
            # padding gExpr
            padding = np.full(pad_length, self.pad_Expr_value,dtype=gExpr.dtype)
            gExpr = np.concatenate([gExpr, padding])
            # padding gExpr_mask
            padding = np.full(pad_length, 0, dtype=gExpr_mask.dtype)
            gExpr_mask = np.concatenate([gExpr_mask, padding])
            # padding gStat
            padding = np.full((pad_length, gStat.shape[1]), 0, dtype=gStat.dtype)
            gStat = np.concatenate([gStat, padding], axis=0)
        else:
            # Crop the array to random n elements
            indices = np.random.choice(current_length, self.gene_total_len, replace=False)
            indices = np.sort(indices)
            # cropping gID
            gID = gID[indices]
            # cropping gPos
            gPos = gPos[indices]
            # cropping gExpr
            gExpr = gExpr[indices]
            # cropping gExpr_mask
            gExpr_mask = gExpr_mask[indices]
            # cropping gStat
            gStat = gStat[indices, :]


        # get aPos and aExpr
        aPos = ATAC_data['peak_positions']
        aExpr = ATAC_data['peak_values']

        # random order the peaks
        random_indices = np.random.permutation(len(aPos))
        aPos = aPos[random_indices]
        aExpr = aExpr[random_indices]

        # set mask to aPos
        aPos_mask = (np.random.rand(len(aPos)) < self.aPos_mask_frac).astype(int)
        aExpr_mask = (np.random.rand(len(aExpr)) < self.aExpr_mask_frac).astype(int)

        # padding and cropping ATAC data
        current_length = aPos.shape[0]
        if current_length < self.atac_total_len:
            # Calculate how many zeros to pad
            pad_length = self.atac_total_len - current_length
            # padding aPos
            padding = np.full(pad_length, self.pad_Pos_value, dtype=aPos.dtype)
            aPos = np.concatenate([aPos, padding])
            # padding aPos_mask
            padding = np.full(pad_length, 0, dtype=aPos_mask.dtype)
            aPos_mask = np.concatenate([aPos_mask, padding])
            # padding aExpr
            padding = np.full(pad_length, self.pad_Expr_value, dtype=aExpr.dtype)
            aExpr = np.concatenate([aExpr, padding])
            # padding aExpr_mask
            padding = np.full(pad_length, 0, dtype=aExpr_mask.dtype)
            aExpr_mask = np.concatenate([aExpr_mask, padding])
        else:
            # Crop the array to random n elements
            indices = np.random.choice(current_length, self.atac_total_len, replace=False)
            indices = np.sort(indices)
            # cropping aPos
            aPos = aPos[indices]
            # cropping aPos_mask
            aPos_mask = aPos_mask[indices]
            # cropping aExpr
            aExpr = aExpr[indices]
            # cropping aExpr_mask
            aExpr_mask = aExpr_mask[indices]

        # group data
        RNA_sc = {
            "gID": gID,
            "gPos": gPos,
            "gExpr": gExpr,
            "gExpr_mask": gExpr_mask,
            'gStat':gStat
        }
        ATAC_sc = {
            "aPos": aPos,
            "aPos_mask": aPos_mask,
            "aExpr": aExpr,
            "aExpr_mask": aExpr_mask
        }
        return RNA_sc, ATAC_sc

    @staticmethod
    def Get_h5ad_info(file_path,cell_index):
        """
        This function is used to extract the expression data of the cell with cell_index in that file_path
        """
        with h5py.File(file_path, 'r') as f:
            # Access the CSR components from the "X" group
            X_group = f['X']
            data = X_group['data']
            indices = X_group['indices']
            indptr = X_group['indptr']

            # Get the nonzero entries for the specified cell
            # a csr matrix is concatenated into a long list by its rows without 0.
            # the indptr is a n_cell+1 list, where the value under index i and i+1 indicates a cell's start idx and end idx
            # for example, if i and i+1 elements in indptr is 100 and 200, respecitvely,
            # then it means the ith cell's data is from data list[100] to data list[199]
            start = indptr[cell_index]
            end = indptr[cell_index + 1]
            cell_data = data[start:end]
            cell_feature_indices = indices[start:end]

            # Read feature annotations from the "var" group
            var_group = f['var']

            # Feature names are stored in var/_index
            feature_names = var_group['_index'][:]
            feature_names = np.array([
                name.decode('utf-8') if isinstance(name, bytes) else str(name)
                for name in feature_names
            ])

            # Extract the "position" information from var (assuming it's stored as integers)
            positions = var_group['position'][:]  # shape: (num_features,)

            # Extract the statistic information from var
            stat_keys = ["batch_mean","batch_std","batch_min","batch_q25","batch_median","batch_q75","batch_max"]
            gene_stat_dict = {k:var_group[k][:] for k in stat_keys}

            # Get feature type information from var/feature_types
            ft_group = var_group['feature_types']
            codes = ft_group['codes'][:]  # shape: (num_features,)
            categories = ft_group['categories'][:]
            categories = np.array([
                cat.decode('utf-8') if isinstance(cat, bytes) else str(cat)
                for cat in categories
            ])

            # Determine the code for gene expression and peaks.
            gene_code = None
            peak_code = None
            for idx, cat in enumerate(categories):
                if cat.lower() in ["gene expression", "gene", "gene_expr"]:
                    gene_code = idx
                elif cat.lower() in ["peaks", "peak"]:
                    peak_code = idx
            if gene_code is None or peak_code is None:
                raise ValueError("Could not identify gene and peak feature codes based on categories.")

            # For the features present in this cell, determine which are genes and which are peaks.
            cell_codes = codes[cell_feature_indices]
            gene_mask = (cell_codes == gene_code)
            peak_mask = (cell_codes == peak_code)

            gene_feature_indices = cell_feature_indices[gene_mask]
            gene_values = cell_data[gene_mask]
            peak_feature_indices = cell_feature_indices[peak_mask]
            peak_values = cell_data[peak_mask]

            # Map feature indices to feature names and positions
            gene_symbols = feature_names[gene_feature_indices]
            gene_positions = positions[gene_feature_indices]
            peak_symbols = feature_names[peak_feature_indices]
            peak_positions = positions[peak_feature_indices]
            gene_stat_matrix = np.column_stack([gene_stat_dict[k][gene_feature_indices] for k in stat_keys])

            RNA_data = {'gene_symbols': gene_symbols,
                        'gene_positions':gene_positions,
                        'gene_values':gene_values,
                        'gene_stat':gene_stat_matrix
                        }
            ATAC_data = {'peak_symbols': peak_symbols,
                         'peak_positions':peak_positions,
                         'peak_values':peak_values
                         }

            return RNA_data, ATAC_data




# test dataset
if __name__ == '__main__':
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
    #
    # RNA_sample, ATAC_sample = MyDataset[123456]
    #
    # RNA_sample['gID']
    # RNA_sample['gPos']
    # RNA_sample['gExpr']
    # RNA_sample['gExpr_mask']
    #
    # ATAC_sample['aPos']
    # ATAC_sample['aPos_mask']
    # ATAC_sample['aExpr']
    # ATAC_sample['aExpr_mask']
    #
    # MyDataset.exprot_drawableidx(export_path = "./references/", val_frac=0.1)


    #stat
    MyDataset = GetData_LLM_stat(pad_id=2,
                                 pad_Pos_value=0,
                                 pad_Expr_value=0,
                                 gene_total_len=3072,
                                 atac_total_len=6144,
                                 gExpr_mask_frac=0.15,  # mask fraction of gExpr
                                 aPos_mask_frac=0,  # mask fraction of aPos
                                 aExpr_mask_frac=0.15,  # mask fraction of aExpr
                                 gene_vocab_path="./references/gene_vocab_single_Vqiuping.json",
                                 Database_dir="/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_MultiomicsData_1M/clean2/")

    temp1 = MyDataset.all_h5ad_path
    len(MyDataset)

    path, id = MyDataset.map_totalIDX_to_localIDX(10329)

    temp0 = [x - 1 for x in MyDataset.h5ad_cellnum_accumulate]
    for i in temp0:
        RNA_data, ATAC_data = MyDataset[i]
        path, id = MyDataset.map_totalIDX_to_localIDX(i)
        RNA_data, ATAC_data = MyDataset.Get_h5ad_info(MyDataset.all_h5ad_path[path], id)
        if np.isnan(RNA_data['gene_stat']).any():
            print(f"{i} has NaN")

    RNA_sample, ATAC_sample = MyDataset[123456]

    RNA_sample['gID']
    RNA_sample['gPos']
    RNA_sample['gExpr']
    RNA_sample['gExpr_mask']
    RNA_sample['gStat']

    ATAC_sample['aPos']
    ATAC_sample['aPos_mask']
    ATAC_sample['aExpr']
    ATAC_sample['aExpr_mask']

    # check position
    temp0 = [x - 1 for x in MyDataset.h5ad_cellnum_accumulate]
    for i in temp0:
        path, id = MyDataset.map_totalIDX_to_localIDX(i)
        RNA_data, ATAC_data = MyDataset.Get_h5ad_info(MyDataset.all_h5ad_path[path], id)
        if (RNA_data['gene_positions']<0).any():
            print(f"{i} has minus values")

    temp0 = [x - 1 for x in MyDataset.h5ad_cellnum_accumulate]
    for i in temp0:
        RNA_data, ATAC_data = MyDataset[i]
        if (RNA_data['gPos']<0).any():
            print(f"{i} has minus values")

    temp0 = np.random.choice(np.arange(len(MyDataset)), size=100, replace=False)
    for i in temp0:
        RNA_data, ATAC_data = MyDataset[i]
        if (RNA_data['gPos']<0).any():
            print(f"{i} has minus values")

    # check position types
    temp0 = [x - 1 for x in MyDataset.h5ad_cellnum_accumulate]
    for i in temp0:
        path, id = MyDataset.map_totalIDX_to_localIDX(i)
        RNA_data, ATAC_data = MyDataset.Get_h5ad_info(MyDataset.all_h5ad_path[path], id)
        print(f"{i}, RNA pos:{RNA_data['gene_positions'].dtype}, ATAC pos: {ATAC_data['peak_positions'].dtype}")

    temp0 = [x - 1 for x in MyDataset.h5ad_cellnum_accumulate]
    for i in temp0:
        RNA_data, ATAC_data = MyDataset[i]
        print(f"{i}, RNA pos:{RNA_data['gPos'].dtype}, ATAC pos: {ATAC_data['aPos'].dtype}")


    # check position max
    temp0 = [x - 1 for x in MyDataset.h5ad_cellnum_accumulate]
    for i in temp0:
        path, id = MyDataset.map_totalIDX_to_localIDX(i)
        RNA_data, ATAC_data = MyDataset.Get_h5ad_info(MyDataset.all_h5ad_path[path], id)
        print(f"{i}, RNA pos:{RNA_data['gene_positions'].dtype}, ATAC pos: {ATAC_data['peak_positions'].dtype}")

    temp0 = [x - 1 for x in MyDataset.h5ad_cellnum_accumulate]
    for i in temp0:
        RNA_data, ATAC_data = MyDataset[i]
        print(f"{i}, RNA pos:{RNA_data['gPos'].dtype}, ATAC pos: {ATAC_data['aPos'].dtype}")