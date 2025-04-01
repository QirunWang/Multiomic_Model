
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import json


def ClassifyGenes(csv_path):
    """
    从CSV文件中读取基因描述数据，并将基因分类为：
    - 无描述（需随机初始化）
    - 重复描述（需随机初始化）
    - 独特描述（可用于Qwen）

    参数:
        csv_path (str): CSV文件路径，需包含 'Symbol' 和 'Summary' 两列

    返回:
        no_desc_genes (List[str]): 无描述的基因Symbol列表
        repeated_desc_genes (List[str]): 重复描述的基因Symbol列表
        unique_desc_dict (Dict[str, str]): 独特描述的基因字典 {Symbol: Summary}
    """

    df = pd.read_csv(csv_path, usecols=["Symbol", "Summary"])
    df = df.dropna(subset=["Symbol"])
    df["Summary"] = df["Summary"].fillna("")

    # 分类处理
    no_desc_df = df[df["Summary"] == ""]
    desc_counts = df[df["Summary"] != ""]["Summary"].value_counts()

    repeated_desc_df = df[
        (df["Summary"] != "") & df["Summary"].isin(desc_counts[desc_counts > 1].index)
    ]

    unique_desc_df = df[
        (df["Summary"] != "") & df["Summary"].isin(desc_counts[desc_counts == 1].index)
    ]

    # 输出结构
    no_desc_genes = no_desc_df["Symbol"].unique().tolist()
    repeated_desc_genes = repeated_desc_df["Symbol"].unique().tolist()
    unique_desc_dict = dict(zip(unique_desc_df["Symbol"], unique_desc_df["Summary"]))
    
    pd.DataFrame({"Symbol": no_desc_genes}).to_csv("no_desc_genes.csv", index=False)
    pd.DataFrame({"Symbol": repeated_desc_genes}).to_csv("repeated_desc_genes.csv", index=False)
    pd.DataFrame(list(unique_desc_dict.items()), columns=["Symbol", "Summary"]).to_csv("unique_desc_genes.csv", index=False)
    

    return no_desc_genes, repeated_desc_genes, unique_desc_dict


class RandomInitializer:
    def __init__(self,
                 n_batch,             
                 gene_len,            
                 gene_total_len,      
                 gExpr_mask_frac,
                 max_gene_num,   
                 max_expr,        
                 max_pos,         
                 embedding_dim,   
                 seed):            

        self.n_batch = n_batch
        self.gene_len = gene_len
        self.gene_total_len = gene_total_len
        self.gExpr_mask_frac = gExpr_mask_frac
        self.max_gene_num = max_gene_num
        self.max_expr = max_expr
        self.max_pos = int(max_pos)
        self.embedding_dim = embedding_dim

        torch.manual_seed(seed)

    def _pad_or_crop(self, tensor, total_len):
        current_len = tensor.size(0)
        if current_len < total_len:
            pad_tensor = torch.zeros(total_len - current_len, dtype=tensor.dtype)
            tensor = torch.cat([tensor, pad_tensor], dim=0)
        else:
            tensor = tensor[:total_len]
        return tensor

    def initialize_embeddings(self, gene_symbols):
        """
        为传入的 symbol 列表生成随机 embedding
        :param gene_symbols: 需要初始化的基因符号列表（去重）
        :return: dict {symbol: torch.Tensor}
        """
        # 为每个 symbol 生成 embedding
        embedding_dict = {
            symbol: torch.randn(self.embedding_dim) for symbol in gene_symbols
        }

        # 构造随机 RNA_module（每次 batch 取 gene_len 个基因）
        batch_gID = []
        batch_gPos = []
        batch_gExpr = []
        batch_gExpr_mask = []
        
        symbol_list = list(gene_symbols)
        total_gene_num = len(symbol_list)

        for _ in range(self.n_batch):
            if self.gene_len > total_gene_num:
                raise ValueError("gene_len 不能大于待初始化的基因总数")

         # 随机选择 gene_len 个 symbol
            selected = torch.randperm(total_gene_num)[:self.gene_len]
            gID_symbols = [symbol_list[i] for i in selected]
            gID_indices = torch.tensor(selected, dtype=torch.int64)
            gID = self._pad_or_crop(gID_indices, self.gene_total_len)

            # 随机位置和表达值
            gPos = self._pad_or_crop(torch.randint(1, self.max_pos, (self.gene_len,), dtype=torch.int64), self.gene_total_len)
            gExpr = self._pad_or_crop(torch.rand(self.gene_len) * self.max_expr, self.gene_total_len)

            gExpr_mask = self._pad_or_crop((torch.rand(self.gene_len) < self.gExpr_mask_frac).to(torch.int64), self.gene_total_len)

            batch_gID.append(gID)
            batch_gPos.append(gPos)
            batch_gExpr.append(gExpr)
            batch_gExpr_mask.append(gExpr_mask)
            
        # Stack tensors
        RNA_module = {
            "gID": torch.stack(batch_gID),            # shape: [n_batch, gene_total_len]
            "gPos": torch.stack(batch_gPos),
            "gExpr": torch.stack(batch_gExpr),
            "gExpr_mask": torch.stack(batch_gExpr_mask),
        }


        return embedding_dict, RNA_module



class QwenInitializer:
    def __init__(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu", embedding_dim=896):
        """
        初始化 Qwen 模型调用器（从本地加载）
        参数:
            model_path (str): 模型路径
            device (str): 使用的设备
            embedding_dim (int): 嵌入向量维度（应与模型输出一致）
        """
        self.device = device
        self.embedding_dim = embedding_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.qwen_model.eval()

        # 冻结权重
        for param in self.qwen_model.parameters():
            param.requires_grad = False

    
    def initialize_embeddings(self, unique_desc_dict, gene_vocab_path):
        """
        用 Qwen 获取每个描述的 embedding，使用 mean pooling，并映射 symbol 为 gene ID。
        
        参数:
            unique_desc_dict (Dict[str, str]): {symbol: description}
            gene_vocab_path (str): json 文件路径，记录 {symbol: id}
        
        返回:
            gene_embedding_dict: {gene_id: mean_pooled_embedding}
        """

        # 1. 读取 gene vocab json
        with open(gene_vocab_path, "r") as f:
            symbol_to_id = json.load(f)

        gene_embedding_dict = {}

        for symbol, desc in unique_desc_dict.items():
            if symbol not in symbol_to_id:
                print("Not found:", symbol)
                continue  # 跳过不在词表中的 symbol
            
            # 2. 编码输入
            inputs = self.tokenizer(desc, return_tensors="pt", truncation=True, max_length=512).to(self.device)

            with torch.no_grad():
                outputs = self.qwen_model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            
            # 3. mean pooling（去除 padding，注意 attention_mask）
            attention_mask = inputs["attention_mask"]
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()  # [1, seq_len, dim]
            masked_hidden = hidden_states * mask
            summed = masked_hidden.sum(dim=1)  # sum over tokens
            counts = mask.sum(dim=1)           # count of valid tokens
            mean_pooled = summed / counts      # [1, hidden_dim]

            # 4. 存储映射（symbol → id → embedding）
            gene_id = symbol_to_id[symbol]
            gene_embedding_dict[gene_id] = mean_pooled.squeeze(0).cpu()  # 去掉 batch 维度

            # 输出 shape
            print(f"Symbol: {symbol} | Gene ID: {gene_id} | Embedding shape: {mean_pooled.shape}")

        return gene_embedding_dict


if __name__ == "__main__":
    
    csv_path = "/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/references/NCBI_summary_From_pyproj2.csv"  
    no_desc_genes, repeated_desc_genes, unique_desc_dict = ClassifyGenes(csv_path)
    
    qwen_path = "/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/qwen_model/qwen"
    qwen_init = QwenInitializer(model_path=qwen_path, embedding_dim=896)
    
    gene_vocab_path = "/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/references/gene_vocab_single_Vqiuping.json"
    qwen_dict = qwen_init.initialize_embeddings(unique_desc_dict, gene_vocab_path)

     # 合并需要随机初始化的 gene 列表
    genes_to_init = list(set(no_desc_genes + repeated_desc_genes))

    # 创建初始化器实例
    random_initializer = RandomInitializer(
        n_batch=3,           
        gene_len=10,           
        gene_total_len=10,     
        gExpr_mask_frac=0.15,  
        max_gene_num=1000,     
        max_expr=10,          
        max_pos=1e6,          
        embedding_dim=896,    
        seed=42               
    )

    # 调用随机初始化函数
    embedding_dict, RNA_module = random_initializer.initialize_embeddings(genes_to_init)
    
    qwen_init = QwenInitializer(embedding_dim=896)
    qwen_dict = qwen_init.initialize_embeddings(unique_desc_dict)

    # 示例输出
    print(f"已为 {len(embedding_dict)} 个基因完成随机初始化。")
    print("RNA_module keys:", RNA_module.keys())
    print("Random 示例 gID:",RNA_module["gID"][0])  # 打印第一个样本的gID索引
    print("Random 示例 embedding 向量:", embedding_dict[genes_to_init[0]])
