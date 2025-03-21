import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
import os
import sys
sys.path.append("./")
from utils.log_manager import LogManager
from scipy import sparse
from multiprocessing import Pool, cpu_count


class multiomeHelper:

    def __init__(self, adata=None, data_path=None, gtf_file=None, metadata=None, n_cores=None):
        """
        初始化 multiomeHelper 类。

        参数：
            adata: 现有的 AnnData 对象（可选）。
            data_path: 数据文件路径（可选）。
            gtf_file: GTF 文件路径（可选）。
            metadata: 元数据字典，包含以下字段（可选）：
                - Species: 物种（例如：人类、小鼠）。
                - Tissue: 组织（例如：PBMC、肝脏）。
                - Celltype: 细胞类型（例如：T细胞、B细胞）。
                - Disease: 疾病状态（例如：健康、癌症）。
                - Technology: 技术（例如：10x 多组学）。
                - Batch: 批次（例如：批次1、批次2）。
        """

        self.logger = LogManager().logger
        self.adata = adata
        self.data_path = data_path
        self.gtf_file = gtf_file
        self.gene_dict = {}  # Initialize gene_dict here
        self.n_cores = n_cores  # 用户指定的核心数

        # Load the data if a data path is provided
        if self.adata is None:
            self.read_data()
        # Load the GTF file if provided
        if gtf_file:
            self.load_gtf()
        # Load the metadata if provided
        if metadata is not None:
            self.add_meta_data(metadata)

    def read_data(self):
        """
        Read the 10x multiome data from the given path.
        This assumes the data consists of two matrices: one for gene expression (RNA) and one for chromatin accessibility (ATAC).
        """
        if self.data_path is None:
            self.logger.error("Data path is not provided.")
            return

        # Corrected extension check
        if self.data_path.endswith(".h5ad"):
            self.adata = sc.read_h5ad(self.data_path)

        elif self.data_path.endswith(".h5"):
            self.adata = sc.read_10x_h5(self.data_path, gex_only=False)

        else:
            self.logger.error("Unsupported file extension. Expected .h5ad or .h5")
            return

        # Print the structure of the combined AnnData object to debug
        print(f"Combined AnnData object shape: {self.adata.shape}")
        print(f"Combined AnnData object var head:\n{self.adata.var.head()}")

        self.logger.info(f"Data loaded successfully from {self.data_path}")

    def calculate_qc_metrics(self):
        """
        Calculate quality control metrics for the multiome data, including
        total counts, number of genes, and other relevant metrics.
        """
        # Debugging: Check the contents of self.adata.var
        print("Features:")
        print(self.adata.var.head())

        # Ensure 'feature_types' column exists and has expected values
        if "feature_types" not in self.adata.var.columns:
            self.logger.error("'feature_types' column is missing in self.adata.var")
            return

        feature_types = self.adata.var["feature_types"]
        unique_feature_types = set(feature_types.unique())
        if "Gene Expression" not in unique_feature_types:
            self.logger.error("'Gene Expression' not found in feature_types. Found: {}".format(unique_feature_types))
            return

        # Filter and calculate QC metrics
        gene_expression_bool = self.adata.var["feature_types"] == "Gene Expression"
        print(f"Boolean index shape: {gene_expression_bool.shape}")
        print(f"AnnData object shape: {self.adata.shape}")

        # Ensure boolean index matches AnnData object shape
        if gene_expression_bool.shape[0] != self.adata.shape[1]:
            self.logger.error("Boolean index does not match AnnData's shape along this dimension.")
            return

        gene_expression_data = self.adata[:, gene_expression_bool]
        sc.pp.calculate_qc_metrics(gene_expression_data, inplace=True)
        self.logger.info("Quality control metrics calculated.")

    def load_gtf(self):
        """
        加载 GTF 文件并提取基因信息。
        """
        if self.gtf_file is None:
            self.logger.error("No GTF file provided for gene annotations.")
            return

        # 加载 GTF 文件并过滤基因
        gtf = pd.read_csv(self.gtf_file, sep='\t', comment='#', header=None)
        genes = gtf[gtf[2] == 'gene']

        # 提取基因信息
        gene_df = genes[[0, 3, 4, 6, 8]].copy()
        gene_df.columns = ['chrom', 'start', 'end', 'strand', 'info']

        # 使用正则表达式提取 gene_id
        gene_df['gene_id'] = gene_df['info'].str.extract(r'gene_id=([^;]+)')

        # 检查 gene_id 是否成功提取
        if gene_df['gene_id'].isna().all():
            self.logger.error("Failed to extract gene_id from GTF file.")
            return

        # 计算 TSS 位置
        gene_df['tss'] = np.where(gene_df['strand'] == '+', gene_df['start'], gene_df['end'])

        # 按染色体分组
        self.gene_dict = {str(chrom): group.set_index('tss') for chrom, group in gene_df.groupby('chrom')}
        self.logger.info(f"GTF file processed and genes grouped by chromosome.")

        # get chrom length and shift length
        temp3 = gtf.loc[gtf[2] == 'chromosome', [0, 4, ]].copy()
        temp3.reset_index(drop=True, inplace=True)
        temp3.columns = ['chrom', 'length']
        temp3['chrom'] = temp3['chrom'].astype(str)
        temp3['length'] = temp3['length'].astype(int)
        temp3['order'] = temp3['chrom']
        temp3.loc[temp3['order']=="X",'order'] = '23'
        temp3.loc[temp3['order'] == "Y",'order'] = '24'
        temp3.loc[temp3['order']=="MT",'order'] = '25'
        temp3['order'] = temp3['order'].astype(int)
        temp3 = temp3.sort_values(by='order', ascending=True)
        temp4 = np.array(temp3['length']).astype(int)
        temp3['length_shift'] = [sum(temp4[range(k),]) for k in range(len(temp4))]
        self.chr_len = temp3
        self.logger.info(f"Chromosome length obtained from GTF.")

    def link_peak_genes(self):
        """
        基于 TSS 位置将染色质可及性峰（来自 ATAC）与基因（来自 GTF 文件）关联。
        将峰与所有基因的关联信息保存到 `uns['gene-peak distance matrix']` 中。
        """
        if not self.gene_dict:
            self.logger.error("基因字典为空，请先加载 GTF 文件。")
            return

        # 假设 `self.adata` 中的峰数据存储在 `var['peak_id']` 中
        atac_peaks = self.adata.var[self.adata.var["feature_types"].isin(["Peaks"])].copy()

        # 使用向量化操作解析 peak ID
        peak_info = atac_peaks['gene_ids'].str.extract(r'([^:]+):(\d+)-(\d+)')
        peak_info.columns = ['chrom', 'start', 'end']
        peak_info['start'] = peak_info['start'].astype(int)
        peak_info['end'] = peak_info['end'].astype(int)


        # 删除解析失败的行
        atac_peaks = pd.concat([atac_peaks, peak_info], axis=1)
        atac_peaks.dropna(subset=['chrom', 'start', 'end'], inplace=True)

        # 将染色体列转换为字符串类型
        atac_peaks['chrom'] = atac_peaks['chrom'].astype(str)

        # 使用用户指定的核心数量，如果未指定则使用默认值
        if self.n_cores is None:
            num_processes = cpu_count() // 4  # 默认使用 1/4 的 CPU 核心
        else:
            num_processes = self.n_cores

        # 使用多进程并行查找所有基因
        with Pool(processes=num_processes) as pool:
            peak_gene_links = pool.map(self.peak_gene_distance, [row for _, row in atac_peaks.iterrows()])

        # 将 peak_gene_links 转换为 DataFrame
        peak_gene_data = []
        for peak_idx, gene_links in enumerate(peak_gene_links):
            for link in gene_links:
                peak_gene_data.append({
                    'peak_chrom': atac_peaks.iloc[peak_idx]['chrom'],
                    'peak_start': atac_peaks.iloc[peak_idx]['start'],
                    'peak_end': atac_peaks.iloc[peak_idx]['end'],
                    'gene_chrom': link['gene_chrom'],
                    'gene_tss': link['gene_tss'],
                    'gene_strand': link['gene_strand'],
                    'gene_id': link['gene_id'],
                    'distance': link['distance']
                })

        peak_gene_df = pd.DataFrame(peak_gene_data)

        # 将峰-基因关联信息保存到 AnnData 对象中
        self.adata.uns['gene-peak distance matrix'] = peak_gene_df
        self.logger.info(f"成功将 {len(peak_gene_df)} 个峰与基因关联。")

    def process_batch(self, batch):
        """
        处理一批峰，找到最近的基因。
        """
        return [self.peak_gene_distance(row) for _, row in batch.iterrows()]

    def peak_gene_distance(self, peak_row):
        """
        找到峰与所有基因的关联信息，并返回一个包含所有基因的列表。
        """
        peak_chrom, peak_start, peak_end = peak_row[['chrom', 'start', 'end']]

        # 获取同一染色体上的基因
        chr_genes = self.gene_dict.get(peak_chrom[-1])
        if chr_genes is None:
            return []

        # 计算峰的中点
        peak_midpoint = (peak_start + peak_end) / 2

        # 获取所有基因的 TSS 和基因信息
        gene_info = []
        for tss, gene_row in chr_genes.iterrows():
            distance = abs(tss - peak_midpoint)
            gene_info.append({
                'peak_chrom': peak_chrom,
                'peak_start': peak_start,
                'peak_end': peak_end,
                'gene_chrom': gene_row['chrom'],
                'gene_tss': tss,
                'gene_strand': gene_row['strand'],
                'gene_id': gene_row['gene_id'],
                'distance': distance
            })

        return gene_info

    def peak_closest_gene(self):
        """
        格式化 AnnData 对象，将距离在 ±3000 bp 范围内的最近基因的基因 ID 保存到 `var['gene_annotation']` 中。
        """
        if 'gene-peak distance matrix' not in self.adata.uns:
            self.logger.error("未找到 'gene-peak distance matrix'，请先运行 `link_peak_genes` 方法。")
            return

        # 获取峰与基因的关联信息
        peak_gene_df = self.adata.uns['gene-peak distance matrix']

        # 过滤掉距离超过 ±3000 bp 的关联信息
        peak_gene_df_filtered = peak_gene_df[peak_gene_df['distance'] <= 3000]

        # 如果没有符合条件的关联信息，记录日志并返回
        if peak_gene_df_filtered.empty:
            self.logger.warning("没有找到距离在 ±3000 bp 范围内的峰-基因关联信息。")
            return

        # 找到每个峰最近的基因
        closest_genes = peak_gene_df_filtered.loc[
            peak_gene_df_filtered.groupby(['peak_chrom', 'peak_start', 'peak_end'])['distance'].idxmin()]

        # 将最近基因的基因 ID 保存到 `var['gene_annotation']` 中
        self.adata.var['gene_annotation'] = None  # 初始化列
        for _, row in closest_genes.iterrows():
            peak_id = f"{row['peak_chrom']}:{row['peak_start']}-{row['peak_end']}"
            gene_annotation = row['gene_id']
            if peak_id in self.adata.var.index:
                self.adata.var.at[peak_id, 'gene_annotation'] = gene_annotation

        self.logger.info("距离在 ±3000 bp 范围内的最近基因的基因 ID 已保存到 `var['gene_annotation']` 中。")

    def create_gene_activity_matrix(self):
        """
        创建一个 Cell*Gene activity matrix，其中每个细胞的峰计数被汇总到对应的基因。
        """
        if 'gene-peak distance matrix' not in self.adata.uns:
            self.logger.error("未找到 'gene-peak distance matrix'，请先运行 `link_peak_genes` 方法。")
            return

        # 获取峰与基因的关联信息
        peak_gene_df = self.adata.uns['gene-peak distance matrix']

        # 获取峰计数矩阵（假设 ATAC 数据存储在 adata.X 中）
        if sparse.issparse(self.adata.X):
            peak_counts = self.adata.X.toarray()  # 将稀疏矩阵转换为稠密矩阵
        else:
            peak_counts = self.adata.X

        # 创建一个字典，存储每个基因对应的峰索引
        gene_to_peaks = {}
        for _, row in peak_gene_df.iterrows():
            peak_id = f"{row['peak_chrom']}:{row['peak_start']}-{row['peak_end']}"
            gene_id = row['gene_id']
            if gene_id not in gene_to_peaks:
                gene_to_peaks[gene_id] = []
            gene_to_peaks[gene_id].append(peak_id)

        # 创建一个空的基因活动矩阵
        num_cells = self.adata.shape[0]
        num_genes = len(gene_to_peaks)
        gene_activity_matrix = np.zeros((num_cells, num_genes), dtype=np.float32)

        # 将峰计数汇总到对应的基因
        for gene_idx, (gene_id, peak_ids) in enumerate(gene_to_peaks.items()):
            peak_indices = [self.adata.var.index.get_loc(peak_id) for peak_id in peak_ids if
                            peak_id in self.adata.var.index]
            if peak_indices:
                gene_activity_matrix[:, gene_idx] = peak_counts[:, peak_indices].sum(axis=1)

        # 创建新的 AnnData 对象
        gene_activity_adata = AnnData(
            X=gene_activity_matrix,
            obs=self.adata.obs.copy(),  # 保留细胞的元数据
            var=pd.DataFrame(index=list(gene_to_peaks.keys()))  # 基因作为列
        )

        # 将基因活动矩阵保存到 uns 中
        self.adata.uns['gene_activity_matrix'] = gene_activity_adata
        self.logger.info("成功创建 Cell*Gene activity matrix。")

    def add_meta_data(self, metadata=None):
        """
        添加元数据到 AnnData 对象中。

        参数：
            metadata: 元数据字典，包含以下字段（可选）：
                - Species: 物种（例如：人类、小鼠）。
                - Tissue: 组织（例如：PBMC、肝脏）。
                - Celltype: 细胞类型（例如：T细胞、B细胞）。
                - Disease: 疾病状态（例如：健康、癌症）。
                - Technology: 技术（例如：10x 多组学）。
                - Batch: 批次（例如：批次1、批次2）。
        """
        if metadata is None:
            print("请为以下元数据字段输入值（按 Enter 跳过）：")

            # 定义元数据字段
            metadata_fields = {
                'Species': '物种（例如：人类、小鼠）',
                'Tissue': '组织（例如：PBMC、肝脏）',
                'Celltype': '细胞类型（例如：T细胞、B细胞）',
                'Disease': '疾病状态（例如：健康、癌症）',
                'Technology': '技术（例如：10x 多组学）',
                'Batch': '批次（例如：批次1、批次2）'
            }

            # 获取用户输入
            metadata = {}
            for field, description in metadata_fields.items():
                value = input(f"{description}: ")
                if value.strip():  # 如果用户输入了值
                    metadata[field] = value
                else:  # 如果用户跳过
                    metadata[field] = '未知'

        # 将元数据添加到 AnnData 对象中
        for field, value in metadata.items():
            self.adata.obs[field] = value

        self.logger.info("元数据已添加到 AnnData 对象中。")

    def format_gene_id(self):
        """
        Format the gene IDs to ensure consistency (e.g., gene symbols, Ensembl IDs, etc.).
        This may involve cleaning or standardizing gene identifiers.
        """
        # For example, clean up gene symbols
        self.adata.var['gene_ids'] = self.adata.var['gene_ids'].str.upper()  # Convert to upper case for consistency
        self.logger.info("Gene IDs formatted.")