import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from accelerate import Accelerator
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import math
from datasets import load_dataset
# from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from GeneATACConfig import GeneATACConfig

# 加载模型和分词器
# model_path = "./qwen_model/qwen"
# tokenizer_path = "./qwen_model/qwen"

# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)


class EmbeddingLNorm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_id=None, init_weights=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.norm = nn.LayerNorm(embedding_dim)

        # If custom_weights is provided, initialize the embedding weights
        if init_weights is not None:
            if init_weights.shape != (vocab_size, embedding_dim):
                raise ValueError("custom_weights must have shape (vocab_size, embedding_dim)")
            with torch.no_grad():
                self.emb.weight.copy_(init_weights)
                # If using a padding_idx, you might want to zero its weights:
                if pad_id is not None:
                    self.emb.weight[pad_id].fill_(0)

    def forward(self, x):
        """
        x: [batch_size, seq_len]
        """
        emb = self.emb(x)       # [bsz, seq_len, embedding_dim]
        emb = self.norm(emb)    # [bsz, seq_len, embedding_dim]
        return emb


class MLP(nn.Module):
    def __init__(self, input_dim, adapter_dim, output_dim):
        super().__init__()
        self.up = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.down = nn.Linear(adapter_dim, output_dim)

    def forward(self, x):
        """
        x: [btz, seq_len, input_dim]
        """
        x = self.up(x)
        x = self.activation(x)
        x = self.down(x)

        return x


class CrossAttentionFFN(nn.Module):
    def __init__(self, embd, num_heads, ffn_hidden_dim=None, dropout=0.1, batch_first = True):
        """
        Args:
            embd (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ffn_hidden_dim (int, optional): Hidden dimension in the feedforward network.
                                             Defaults to 4 * embd if not provided.
            dropout (float): Dropout probability.
        """
        super().__init__()
        # MultiheadAttention with batch_first=True expects inputs as [batch, seq_len, embd]
        self.attn = nn.MultiheadAttention(embed_dim=embd, num_heads=num_heads, batch_first=batch_first)

        # Set the feedforward hidden dimension (default: 4x embedding dimension)
        ffn_hidden_dim = ffn_hidden_dim or (embd * 4)
        self.ffn = nn.Sequential(
            nn.Linear(embd, ffn_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embd),
            nn.Dropout(dropout)
        )
        # Layer norms for residual connections
        self.norm1 = nn.LayerNorm(embd)
        self.norm2 = nn.LayerNorm(embd)

    def forward(self, query, key, value):
        """
        Args:
            query (Tensor): Tensor of shape [btz, seq_len_query, embd] (typically from the decoder).
            key_value (Tensor): Tensor of shape [btz, seq_len_kv, embd] (typically from the encoder).

        Returns:
            Tensor: Output tensor of shape [btz, seq_len_query, embd].
        """
        # Perform cross-attention: output aligns with the query's shape.
        attn_out, _ = self.attn(query=query, key=key, value=value)
        # Add & normalize
        attn_out = self.norm1(query + attn_out)
        # Feedforward network with residual connection
        ffn_out = self.ffn(attn_out)
        out = self.norm2(attn_out + ffn_out)
        return out


class LLMadapter_test(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input

class LLM_test(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return input
    
class QWENModel(nn.Module):
    def __init__(self, model_path):
        super(QWENModel, self).__init__()
        # 加载预训练模型
        self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path)
        # 冻结权重
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        # self.embedding_adapter = nn.Linear(1024, 896)

    def forward(self, embeddings):
       # adapted_embeddings = self.embedding_adapter(embeddings)
        try:
            outputs = self.qwen_model(inputs_embeds=embeddings)
        except Exception as e:
            print("Error in model call:", e)
            raise e
        return outputs


class PromptEmbedder(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path)

    
    def encode_text(self, text: str, device=None) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt")
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.qwen_model(**inputs, output_hidden_states=True)
            
        prompt_embeddings = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        return prompt_embeddings
        
        # if mode == 'cls':
        #     return prompt_embeddings[:, 0, :]
        # elif mode == 'last':
        #     return prompt_embeddings[:, -1, :]
        # elif mode == 'mean':
        #     mask = inputs["attention_mask"]
        #     masked = prompt_embeddings * mask.unsqueeze(-1)
        #     return masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        
    def encode_texts(self, texts: list) -> torch.Tensor:
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.qwen_model(**inputs, output_hidden_states=True)
        
        prompt_embeddings = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        return prompt_embeddings

        # if mode == 'cls':
        #     return last_hidden[:, 0, :]
        # elif mode == 'last':
        #     lengths = inputs["attention_mask"].sum(dim=1) - 1
        #     return last_hidden[torch.arange(last_hidden.size(0)), lengths, :]
        # elif mode == 'mean':
        #     mask = inputs["attention_mask"]
        #     masked = last_hidden * mask.unsqueeze(-1)
        #     return masked.sum(dim=1) / mask.sum(dim=1, keepdim=True)
        # else:
        #     raise ValueError("mode must be 'cls', 'last', or 'mean'")

    

# main.py

# def main():
#     model_path = "/path/to/local/qwen"  # 替换为你实际的本地路径
    

#     embedder = PromptEmbedder(model_path)
#     embedding = embedder.encode_prompt(prompt, mode="cls")
#     print("Single prompt embedding shape:", embedding.shape)

#     # 示例：多个 prompt
#     prompts = [
#         "This is the first prompt.",
#         "This is the second.",
#         "Third one here!"
#     ]
#     batch_embeddings = embedder.encode_prompts(prompts, mode="mean")
#     print("Batch prompt embedding shape:", batch_embeddings.shape)

# if __name__ == "__main__":
#     main()




class Multiomics_plus(nn.Module):
    # input
    # Gene ID seq: [g1, ..., gn, pad, ..., pad] [btz,g_len]
    # Gene Pos seq: [p1, ..., pn, 0, ..., 0] [btz,g_len]
    # Gene Expr seq: [x1, ..., xn, 0, ..., 0] [btz,g_len]
    # Gene Expr mask: [0, ..., 1, 0, ..., 0] [btz,g_len]

    # ATAC Pos seq: [p1, ..., pm, 0, ..., 0] [btz,a_len]
    # ATAC Pos mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
    # ATAC Expr seq: [s1, ..., sm, 0, ..., 0] [btz,a_len]
    # ATAC Expr mask: [0, ..., 1, 0, ..., 0] [btz,a_len]

    def __init__(self,
                 gene_vocab_size: int = 19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
                                                # 19069 and 19070 for RNA start and end,
                                                # 19071 and 19072 for ATAC start and end
                 pad_id=0,
                 mask_id=1,
                 cls_id=2,
                 RNAstart_id = 19068,
                 RNAend_id = 19069,
                 ATACstart_id = 19070,
                 ATACend_id = 19071,

                 ebdim_total = 1024,

                 pad_value = 0,  # this is what value of pad token should put in for the masked token
                 mask_value = -1, # this is what value of mask token should put in for the masked token
                 # RNAstart_value=0, # this is what value of pad token should put in for the masked token
                 # RNAend_value=0, # this is what value of pad token should put in for the masked token
                 # ATACstart_value=0, # this is what value of pad token should put in for the masked token
                 # ATACend_value=0, # this is what value of pad token should put in for the masked token

                 nhead: int = 16,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Args:
            emb_dim (int): Common embedding dimension.
            num_heads (int): Number of attention heads.
            g_vocab_size (int): Vocabulary size for gID.
            a_vocab_size (int): Vocabulary size for aPos.
            mask_token_init (float): Initial value for mask token embeddings.
        """
        #####init
        super().__init__()
        self.gene_vocab_size = gene_vocab_size
        self.pad_id = pad_id
        self.mask_id = mask_id
        self.cls_id = cls_id
        self.RNAstart_id = RNAstart_id
        self.RNAend_id = RNAend_id
        self.ATACstart_id = ATACstart_id
        self.ATACend_id = ATACend_id

        self.ebdim_total = ebdim_total

        self.padvalue = pad_value
        self.maskvalue = mask_value

        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        

        ##### define sub-modules whose parameters are needed updating
        # Gene and special token id embedding
        self.ID_emb = EmbeddingLNorm(self.gene_vocab_size, self.ebdim_total, self.pad_id)

        # RNA Expr projection adaptor, used to project one float value to a vector
        self.RNA_Expr_proj = MLP(input_dim=1,
                                  adapter_dim=self.ebdim_total,
                                  output_dim=self.ebdim_total)

        # ATAC Expr projection adaptor, used to project one float value to a vector
        self.ATAC_Expr_proj = MLP(input_dim = 1,
                                  adapter_dim = self.ebdim_total,
                                  output_dim = self.ebdim_total)

        # Self-attention transformer modules (using batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer( d_model=self.ebdim_total,  # 2048
                                                    nhead=self.nhead,
                                                    dim_feedforward=1 * self.ebdim_total,
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=self.num_layers)

        # Adaptor to LLM
        self.LLMadaptor = LLMadapter_test()

        # LLM
        self.LLM = LLM_test()
        # freeze parameters
        for param in self.LLM.parameters():
            param.requires_grad = False

        # Prediction heads with LLM output:
        # aPos_head: predict discrete tokens (logits over a_vocab_size)
        self.aPos_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1)
        # aExpr_head and gExpr_head: predict float values (regression).
        self.aExpr_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1)
        self.gExpr_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1)


    @staticmethod
    def genomic_sinusoidal_encoding(positions, emb_dim):
        """
        Compute sinusoidal positional encodings for genomic positions.

        Args:
            positions (Tensor): Tensor of shape [batch, seq_len] containing genomic positions as integers.
            emb_dim (int): Dimension of the positional encoding.
            scale (float): Factor to normalize the positions (e.g., 1e6).

        Returns:
            Tensor: Positional encodings of shape [batch, seq_len, emb_dim].
        """
        # Normalize positions to reduce the scale.
        normalized_positions = positions.float()  # now in a smaller range
        normalized_positions = normalized_positions.unsqueeze(-1)  # [batch, seq_len, 1]

        # Create a tensor of dimension indices.
        div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32, device=positions.device) * -(math.log(10000.0) / emb_dim))
        # div_term shape is [emb_dim/2]. We need to ensure it broadcasts correctly.

        pe = torch.zeros(normalized_positions.size(0), normalized_positions.size(1), emb_dim, device=positions.device)
        pe[:, :, 0::2] = torch.sin(normalized_positions * div_term)
        pe[:, :, 1::2] = torch.cos(normalized_positions * div_term)
        return pe


    # Get the RNAseq embd by concatenate gID, gPos, and gExpr
    def Embed_RNA(self, module_RNA):
        """
        Encodes RNA features.
        Expects RNA to be a dict with keys:
            # gID: [g1, ..., gn, pad, ..., pad] [btz,g_len]
            # gPos: [p1, ..., pn, 0, ..., 0] [btz,g_len]
            # gExpr: [x1, ..., xn, 0, ..., 0] [btz,g_len]
            # gExpr_mask: [0, ..., 1, 0, ..., 0] [btz,g_len]
        Returns:
          Tensor of shape [batch, g_len, emb_dim].
        """
        gID = module_RNA["gID"]
        gPos = module_RNA["gPos"]
        gExpr = module_RNA["gExpr"]
        gExpr_mask = module_RNA["gExpr_mask"]

        # Embed gID
        gID_embeddings = self.ID_emb(gID)

        # Embed gPos with positional encoding
        gPos_embeddings = self.genomic_sinusoidal_encoding(gPos, self.ebdim_total)
        # make the pad embeddings to value 0
        mask = (gID == self.pad_id).unsqueeze(-1).to(gPos_embeddings.device) # Expand to shape [btz, seq_len, 1]
        gPos_embeddings = gPos_embeddings.masked_fill(mask, self.padvalue)  # make all values of the dims of padding token to 0

        # Embed gExpr
        # make the mask postion to value -1
        mask = (gExpr_mask == 1).to(gExpr.device)
        gExpr = gExpr.masked_fill(mask,self.maskvalue)  # make all values mask position to -1
        # project it to a vector
        gExpr = gExpr.unsqueeze(-1)
        gExpr_embeddings = self.RNA_Expr_proj(gExpr)
        # make the pad embeddings to value 0
        mask = (gID == self.pad_id).unsqueeze(-1).to(gExpr_embeddings.device)
        gExpr_embeddings = gExpr_embeddings.masked_fill(mask,self.padvalue)

        # Sum up all embeddings
        RNA_embeddings = gID_embeddings + gPos_embeddings + gExpr_embeddings

        # Add RNA start and end tokens
        RNAstart_token = torch.full((gID.shape[0], 1), self.RNAstart_id, dtype=torch.long, device=gID.device)
        RNAstart_token_embeddings = self.ID_emb(RNAstart_token)
        RNAend_token = torch.full((gID.shape[0], 1), self.RNAend_id, dtype=torch.long, device=gID.device)
        RNAend_token_embeddings = self.ID_emb(RNAend_token)
        RNA_embeddings = torch.cat((RNAstart_token_embeddings, RNA_embeddings, RNAend_token_embeddings), dim=1)

        return RNA_embeddings


    def Embed_ATAC(self, module_ATAC):
        """
        Encodes ATAC features.
        Expects ATAC to be a dict with keys:
            # aPos: [p1, ..., pn, 0, ..., 0] [btz,a_len]
            # aPos_mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
            # aExpr: [x1, ..., xn, 0, ..., 0] [btz,a_len]
            # aExpr_mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
        Returns:
          Tensor of shape [batch, a_len, emb_dim].
        """
        aPos = module_ATAC["aPos"]
        aPos_mask = module_ATAC["aPos_mask"]
        aExpr = module_ATAC["aExpr"]
        aExpr_mask = module_ATAC["aExpr_mask"]

        # Embed aPos with positional encoding
        # make the mask postion to value -1
        mask = (aPos_mask == 1)
        aPos = aPos.masked_fill(mask, self.maskvalue)  # make all values of the dims of mask position to 0
        # make positional encoding
        aPos_embeddings = self.genomic_sinusoidal_encoding(aPos, self.ebdim_total)
        # make the pad embeddings to value 0
        mask = (aPos == 0).unsqueeze(-1)  # Expand to shape [btz, seq_len, 1]
        aPos_embeddings = aPos_embeddings.masked_fill(mask,self.padvalue)  # make all values of the dims of padding token to 0


        # Embed aExpr
        # make the mask postion to value -1
        mask = (aExpr_mask == 1)
        aExpr = aExpr.masked_fill(mask, self.maskvalue)  # make all values mask position to -1
        # project it to a vector
        aExpr = aExpr.unsqueeze(-1)
        aExpr_embeddings = self.ATAC_Expr_proj(aExpr)
        # make the pad embeddings to value 0
        mask = (aPos == 0).unsqueeze(-1)
        aExpr_embeddings = aExpr_embeddings.masked_fill(mask, self.padvalue)

        # Sum up all embeddings
        ATAC_embeddings = aPos_embeddings + aExpr_embeddings

        # Add ATAC start and end tokens
        ATACstart_token = torch.full((aPos.shape[0], 1), self.ATACstart_id, dtype=torch.long, device=aPos.device)
        ATACstart_token_embeddings = self.ID_emb(ATACstart_token)
        ATACend_token = torch.full((aPos.shape[0], 1), self.ATACend_id, dtype=torch.long, device=aPos.device)
        ATACend_token_embeddings = self.ID_emb(ATACend_token)
        ATAC_embeddings = torch.cat((ATACstart_token_embeddings, ATAC_embeddings, ATACend_token_embeddings), dim=1)

        return ATAC_embeddings


    def task_determine(self,module_RNA, module_ATAC):
        # 1. given unmasked gID and gExpr, all-masked aPos and aExpr, predict the actual aPos;
        # 2. given unmasked gID and gExpr, partially-masked aPos and all-masked aExpr, predict the actual aPos left;
        # 3. given unmasked gID, gExpr, and aPos, and all-masked aExpr, predict the actual aExpr;
        # 4. given unmasked gID, gExpr, aPos, and partially-masked aExpr, predict the actual aExpr left;
        # 5. given unmasked aPos, aExpr, gID, all masked gExpr, predict the actual gExpr;
        # 6. given unmasked aPos, aExpr, gID, partially masked gExpr, predict the actual gExpr left.
        pass


    def forward_beforeLLM(self,module_RNA, module_ATAC, prompt_embedding):
        """
        Args:
          module1 (dict): contains keys "gID", "gExpr" and optionally "gExpr_mask" of shape [batch, g_len]
          module2 (dict): contains keys "aPos", "aExpr" and optionally "aPos_mask", "aExpr_mask" of shape [batch, a_len]
          task (int): task id (1 to 6) indicating which prediction to make.

        Returns:
          For tasks 1-4, predictions are made on module2 outputs (for aPos or aExpr).
          For tasks 5-6, predictions are made on module1 outputs (for gExpr).
          The returned tensor(s) only include predictions for positions that were masked.
        """

        # Encode each module.
        RNA_embeddings = self.Embed_RNA(module_RNA)
        ATAC_embeddings = self.Embed_ATAC(module_ATAC)
        print(RNA_embeddings.shape)
        print(ATAC_embeddings.shape)
        # 修复 batch size 不一致问题
        if prompt_embedding.shape[0] != RNA_embeddings.shape[0]:
            prompt_embedding = prompt_embedding.expand(RNA_embeddings.shape[0], -1, -1)

        # concate sentence
        All_embeddings = torch.concat([prompt_embedding, RNA_embeddings, ATAC_embeddings],dim=1)

        return All_embeddings

    def forward(self,module_RNA, module_ATAC):
        """
        Args:
          module1 (dict): contains keys "gID", "gExpr" and optionally "gExpr_mask" of shape [batch, g_len]
          module2 (dict): contains keys "aPos", "aExpr" and optionally "aPos_mask", "aExpr_mask" of shape [batch, a_len]
          task (int): task id (1 to 6) indicating which prediction to make.

        Returns:
          For tasks 1-4, predictions are made on module2 outputs (for aPos or aExpr).
          For tasks 5-6, predictions are made on module1 outputs (for gExpr).
          The returned tensor(s) only include predictions for positions that were masked.
        """

        # Encode each module.
        RNA_embeddings = self.Embed_RNA(module_RNA)
        ATAC_embeddings = self.Embed_ATAC(module_ATAC)

        # concate sentence
        All_embeddings = torch.concat([RNA_embeddings, ATAC_embeddings],dim=1)

        # transformer block
        All_embeddings = self.transformer_encoder(All_embeddings)

        # Pass through adapter
        All_embeddings = self.LLMadaptor(All_embeddings)

        # Pass through LLM
        All_embeddings = self.LLM(All_embeddings)

        # Pass through decoder
        # devide the RNA and ATAC part
        temp1 = [1,module_RNA['gExpr'].shape[1] ,1 ,1 ,module_ATAC['aExpr'].shape[1],1]
        _, RNA_embeddings, _, _, ATAC_embeddings, _ = torch.split(All_embeddings,temp1, dim=1)
        # predict with decoder
        out_RNA = self.gExpr_head(RNA_embeddings)
        out_ATAC = self.gExpr_head(ATAC_embeddings)


        return out_RNA,out_ATAC




# class GeneATACEmbedding(nn.Module):
#     def __init__(self, config: GeneATACConfig):
#         super().__init__()
#         self.config = config
#         self.gene_embedding = nn.Linear(self.config.gene_input_dim, self.config.embedding_dim)
#         self.atac_embedding = nn.Linear(self.config.atac_input_dim, self.config.embedding_dim)


#     def forward(self, gene_data, atac_data):
#         embedded_gene = self.gene_embedding(gene_data)
#         embedded_atac = self.atac_embedding(atac_data)
#         return embedded_gene, embedded_atac
    

class QWENModel(nn.Module):
    def __init__(self, model_path):
        super(QWENModel, self).__init__()
        # 加载预训练模型
        self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path)
        # 冻结权重
        for param in self.qwen_model.parameters():
            param.requires_grad = False
        # self.embedding_adapter = nn.Linear(1024, 896)

    def forward(self, embeddings):
       # adapted_embeddings = self.embedding_adapter(embeddings)
        try:
            outputs = self.qwen_model(inputs_embeds=embeddings)
        except Exception as e:
            print("Error in model call:", e)
            raise e
        return outputs




