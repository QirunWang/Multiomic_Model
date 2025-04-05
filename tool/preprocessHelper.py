import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from anndata import AnnData
import os
import logging
import json

position_dict_path = '/home/share/huadjyin/home/lishaoshuai/xierunda/'

with open(position_dict_path+"gene_global_position_dict.json", "r") as json_file:
    gene_global_position_dict = json.load(json_file)
with open(position_dict_path+"chrom_prefix_length.json", "r") as json_file:
    chrom_prefix_length = json.load(json_file)

class preprocessHelper:

    def __init__(self, 
                 adata=None, data_path=None, 
                 save_path=None,
                 gene_global_position_dict = gene_global_position_dict, 
                 chrom_prefix_length = chrom_prefix_length, 
                 RNA_scaling_factor=10000, ATAC_scaling_factor=10000):
        """
        初始化 multiomeHelper 类。

        参数：
            adata: 现有的 AnnData 对象（可选）。
            data_path: 数据文件路径（可选）。
        """

        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.adata = adata
        self.data_path = data_path
        self.RNA_scaling_factor = RNA_scaling_factor
        self.ATAC_scaling_factor = ATAC_scaling_factor
        self.gene_global_position_dict = gene_global_position_dict
        self.chrom_prefix_length = chrom_prefix_length
        self.save_path = save_path 
        
        if self.data_path:
            dir_name = os.path.dirname(self.data_path)  # 
            file_name = os.path.basename(self.data_path)  # 
            self.file_name = os.path.join(dir_name, file_name) if dir_name else f"./{file_name}"
        else:
            self.file_name = "Unknown"

        # Load the data if a data path is provided
        if self.adata is None:
            self.read_data()

    def read_data(self):
        """
        Read the 10x multiome data from the given path.
        This assumes the data consists of two matrices: one for gene expression (RNA) and one for chromatin accessibility (ATAC).
        """
        if self.data_path is None:
            self.logger.error(f"{self.file_name}: Data path is not provided.")
            return

        # Corrected extension check
        if self.data_path.endswith(".h5ad"):
            self.adata = sc.read_h5ad(self.data_path)

        elif self.data_path.endswith(".h5"):
            self.adata = sc.read_10x_h5(self.data_path, gex_only=False)

        else:
            self.logger.error(f"{self.file_name}: Unsupported file extension.")
            return

        # Print the structure of the combined AnnData object to debug
        print(f"Combined AnnData object shape: {self.adata.shape}")
        print(f"Combined AnnData object var head:\n{self.adata.var.head()}")

        self.logger.info(f"Data loaded successfully from {self.file_name}")
    
    def filter_by_mito_genes(self, mito_threshold=0.1):
        """
        Filter out cells with high mitochondrial gene expression.
        
        Parameters
        ----------
        mito_threshold: float
            The threshold for mitochondrial gene expression.
        
        Returns
        -------
        adata: AnnData
            The filtered AnnData object.
        """
        gene_expression_bool = self.adata.var["feature_types"] == "Gene Expression"
        gene_expression_data = self.adata[:, gene_expression_bool]
        mito_genes = gene_expression_data.var_names.str.startswith('MT-')
        if mito_genes.any():
            mito_ratio = np.sum(gene_expression_data[:, mito_genes].X, axis=1) / np.sum(gene_expression_data.X, axis=1)
            self.adata.obs['mito_ratio'] = mito_ratio.A1
            self.adata = self.adata[self.adata.obs['mito_ratio'] < mito_threshold, :]
            self.logger.info(f"{self.file_name}:Filtered cells by mitochondrial gene ratio.")
            
        return self.adata
    
    def filter_by_valid_gene(self, min_genes = 100):
        """
        Filter out cells with low gene expression.
        
        Parameters
        ----------
        min_genes: int
            The minimum number of genes required per cell.
        
        Returns
        -------
        adata: AnnData
            The filtered AnnData object.
        """
        gene_expression_bool = self.adata.var["feature_types"] == "Gene Expression"
        gene_expression_data = self.adata[:, gene_expression_bool]
        sc.pp.filter_cells(gene_expression_data, min_genes=min_genes)

        # Apply the filtered cell indices to the original AnnData object
        self.adata = self.adata[gene_expression_data.obs_names, :]
        self.logger.info(f"{self.file_name}:Filtered cells by minimum genes count.")
        
        return self.adata
    
    def remove_mito_genes(self):
        """
        Remove mitochondrial genes from the data.
        
        Returns
        -------
        adata: AnnData
            The filtered AnnData object.
        """
        # Identify genes starting with 'MT-'
        mito_genes = self.adata.var_names.str.startswith('MT-')
        
        # Remove mitochondrial genes
        self.adata = self.adata[:, ~mito_genes]
        self.logger.info(f"{self.file_name}: Removed mitochondrial genes.")
        
    
    def filter_standard_peaks(self):
        """
        Filter out non-standard peaks from the data.
        
        Returns
        -------
        adata: AnnData
            The filtered AnnData object.
        """
        # Filter out non-standard peaks
        is_peak = self.adata.var["feature_types"] == "Peaks"
        standard_chromosomes = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'}
        peak_chromosomes = self.adata.var_names[is_peak].str.split(":").str[0]
        is_standard_peak = peak_chromosomes.isin(standard_chromosomes)
        valid_vars = pd.Series(True, index=self.adata.var_names)
        valid_vars.loc[is_peak] = is_standard_peak
        self.adata = self.adata[:, valid_vars.values]
        self.logger.info(f"{self.file_name}:Filtered standard peaks.")
        
    
    def filter_by_valid_peaks(self, min_peak = 4000):
        """
        Filter out cells with low gene expression.
        
        Parameters
        ----------
        min_peak: int
            The minimum number of peaks required per cell.
        
        Returns
        -------
        adata: AnnData
            The filtered AnnData object.
        """
        peak_bool = self.adata.var["feature_types"] == "Peaks"
        peak_data = self.adata[:, peak_bool]
        sc.pp.filter_cells(peak_data, min_genes=min_peak)

        # Apply the filtered cell indices to the original AnnData object
        self.adata = self.adata[peak_data.obs_names, :]
        self.logger.info(f"{self.file_name}:Filtered cells by minimum peaks count.")

    
    def normalize_and_log1p(self):
        """
        Perform total count normalization followed by log1p transformation on RNA-seq and ATAC data.

        Returns
        -------
        None
        """
        if self.adata is None:
            self.logger.error(f"{self.file_name}: No AnnData object found.")
            return None

        # RNA
        gene_expression_bool = self.adata.var["feature_types"] == "Gene Expression"
        gene_adata = self.adata[:, gene_expression_bool].copy()
        if not gene_expression_bool.any():
            self.logger.warning(f"{self.file_name}: No RNA-seq data found in feature_types.")
        else:
            # Normalization
            sc.pp.normalize_total(gene_adata, target_sum=self.RNA_scaling_factor)
            
            # log1p 
            self.logger.info(f"{self.file_name}: Applying normalization and log1p transformation on RNA-seq data.")
            sc.pp.log1p(gene_adata)
            self.adata.X[:, gene_expression_bool] = gene_adata.X


        # ATAC
        atac_bool = self.adata.var["feature_types"] == "Peaks"
        atac_data = self.adata[:, atac_bool].copy()
        if not atac_bool.any():
            self.logger.warning(f"{self.file_name}: No ATAC data found in feature_types.")
        else:
            # Normalization
            self.logger.info(f"{self.file_name}: Performing total count normalization on ATAC data, construct on dense X")
            sc.pp.normalize_total(atac_data, target_sum=self.ATAC_scaling_factor)

            temp_dense_adata = self.adata.X.toarray()
            temp_dense_adata[:, atac_bool] = atac_data.X.toarray()
            
            self.adata.X = sp.csr_matrix(temp_dense_adata) 
            
        self.logger.info(f"{self.file_name}: Normalization and log1p transformation applied to RNA-seq and ATAC data.")


    
    
    def apply_all_preprocessing(self, mito_threshold=0.1, min_genes=100):
        """
        Apply all preprocessing steps in sequence.
        
        Parameters
        ----------
        mito_threshold: float
            The threshold for mitochondrial gene expression.
        
        min_genes: int
            The minimum number of genes required per cell.
        
        Returns
        -------
        adata: AnnData
            The preprocessed AnnData object.
        """
        # Apply all preprocessing steps
        self.filter_by_mito_genes(mito_threshold=mito_threshold)
        self.filter_by_valid_gene(min_genes=min_genes)
        self.remove_mito_genes()
        self.filter_standard_peaks()
        self.filter_by_valid_peaks()
        self.normalize_and_log1p()
        self.add_gene_global_position()
        self.add_peak_global_positions()
        self.remove_nan()
        

        self.logger.info("All preprocessing steps applied.")
    
    def add_gene_global_position(self):
        """
        Add the global position of genes and peaks to the AnnData object.
        
        """
        #Gene position:
        
        gene_expression_bool = self.adata.var["feature_types"] == "Gene Expression"
        self.adata.var.loc[gene_expression_bool, "position"] = self.adata.var.loc[gene_expression_bool, "gene_ids"].map(self.gene_global_position_dict)
        self.logger.info(f"{self.file_name}: Added global positions for genes.")
        
    def compute_peak_position(self, peak_id):
        """
        计算 peak 的全基因组位置。
        
        参数：
        - peak_id: peak 的标识符，形如 'chr1:1000-2000'
        
        返回：
        - 计算的全基因组位置
        """
        try:
            chrom, coords = peak_id.split(':')  
            start, end = map(int, coords.split('-'))  
            mid_point = (start + end) // 2 
            chrom_id = chrom.replace('chr', '')  
            return self.chrom_prefix_length.get(chrom_id, 0) + mid_point  
        except Exception as e:
            print(f"Error processing peak ID {peak_id}: {e}")
            return np.nan

    
    def add_peak_global_positions(self):
        """
        计算 peaks 的 position 并更新到 adata.var['position']。
        
        参数：
        - peak_bool: 布尔索引，标识 peaks 行。
        """
        # 计算 position 列
        peak_bool = self.adata.var["feature_types"] == "Peaks"
        self.adata.var.loc[peak_bool, 'position'] = self.adata.var.loc[peak_bool, 'gene_ids'].apply(self.compute_peak_position)
        self.logger.info(f"{self.file_name}: Added global positions for peaks.")
                

    def remove_nan(self):
        """
        remove gene and peaks with Nan in Position
        """
        
        non_nan_position = self.adata.var['position'].notna()

        self.adata = self.adata[:, non_nan_position]
        
    def get_adata_size(self):
        """
        Get the shape of the AnnData object.
        
        Returns
        -------
        tuple
            The shape of the AnnData object.
        """
        return self.adata.shape
    
    def save_as_filtered(self):
        """
        Save the filtered data as an .h5ad file in the specified save_path.
        The filename is based on the original file name with '_clean' appended before the extension.
        If save_path is not provided, the file is saved in the current working directory.
        """
        if self.data_path:
            base_name = os.path.basename(self.data_path)
            base_name = os.path.splitext(base_name)[0]
            filtered_filename = f"{base_name}_clean.h5ad"
        else:
            filtered_filename = "filtered_clean.h5ad"
        
        save_dir = self.save_path if self.save_path else os.getcwd()
        filtered_path = os.path.join(save_dir, filtered_filename)

        self.adata.write(filtered_path)
        self.logger.info(f"Filtered data saved to {filtered_path}.")



class H5adSplitter:
    def __init__(self, data_path, save_path, cells_per_split=10000):
        """
        初始化 H5adSplitter 类。
        
        参数：
        data_path: str
            输入数据的路径（.h5ad 文件）。
        save_path: str
            拆分文件保存路径。
        cells_per_split: int
            每个拆分文件包含的最大细胞数，默认为 10000。
        """
        self.data_path = data_path
        self.save_path = save_path
        self.cells_per_split = cells_per_split
        self.adata = None

        # 读取数据
        self.read_data()

    def read_data(self):
        """读取输入的 .h5ad 数据文件"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Input file {self.data_path} not found.")
        
        self.adata = sc.read_h5ad(self.data_path)
        print(f"Loaded data with shape: {self.adata.shape}")
    
    def split_and_save(self):
        """将数据按细胞数量拆分并保存为多个 .h5ad 文件"""
        num_cells = self.adata.shape[0]
        num_splits = (num_cells // self.cells_per_split) + (1 if num_cells % self.cells_per_split != 0 else 0)

        # 获取原始文件名用于保存
        base_name = os.path.basename(self.data_path)
        base_name_no_ext = os.path.splitext(base_name)[0]

        for i in range(num_splits):
            # 计算当前拆分文件的细胞范围
            start_idx = i * self.cells_per_split
            end_idx = min((i + 1) * self.cells_per_split, num_cells)

            # 拆分数据
            adata_split = self.adata[start_idx:end_idx, :]

            # 保存拆分的数据
            split_filename = f"{base_name_no_ext}_{i + 1}.h5ad"
            split_path = os.path.join(self.save_path, split_filename)
            adata_split.write(split_path)
            print(f"Saved split {i + 1} to {split_path}")

        print("Data split and saved successfully.")