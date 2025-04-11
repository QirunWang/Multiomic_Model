#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/03/28 09:24
# @Author  : Kieran Wang
# @File    : model3.py
# @Software: PyCharm


import gc

import numpy as np
from torchinfo import summary
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.optim as optim
import math
import random

import sklearn
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import LoraConfig
from transformers import AutoModelForCausalLM
from peft import get_peft_model

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
    def __init__(self, input_dim, adapter_dim, output_dim,dropout_prob):
        super().__init__()
        self.up = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = None
        self.down = nn.Linear(adapter_dim, output_dim)

    def forward(self, x):
        """
        x: [btz, seq_len, input_dim]
        """
        x = self.up(x)
        x = self.activation(x)
        if self.dropout is not None:
            x = self.dropout(x)
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
    def __init__(self,
                 model_path = '/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/qwen_model/qwen',
                 freeze = True):
        super(QWENModel, self).__init__()
        # 加载预训练模型
        self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path)
        # 冻结权重
        if freeze:
            for param in self.qwen_model.parameters():
                param.requires_grad = False
        # self.embedding_adapter = nn.Linear(1024, 896)

    def forward(self, embeddings, attention_mask):
       # adapted_embeddings = self.embedding_adapter(embeddings)
        try:
            outputs = self.qwen_model(inputs_embeds=embeddings,
                                      attention_mask=attention_mask, # use to filter out padding
                                      output_hidden_states=True)
        except Exception as e:
            print("Error in model call:", e)
            raise e
        return outputs


class PromptEmbedder(nn.Module):
    def __init__(self, model_path = '/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/qwen_model/qwen'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.qwen_model = AutoModelForCausalLM.from_pretrained(model_path)

    def encode_text(self, text: str, device=None) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt")
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.qwen_model(**inputs, output_hidden_states=True)

        # prompt_embeddings = outputs.hidden_states[-1]  # [1, seq_len, hidden_size]
        prompt_embeddings = outputs.hidden_states[-1]  # [1, 1, hidden_size]

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

        #prompt_embeddings = outputs.hidden_states[-1]  # [1, 1, hidden_size]
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


class Multiomics_plus(nn.Module):
    # input
    # RNA module dict:
        # Gene ID seq: [g1, ..., gn, pad, ..., pad] [btz,g_len]
        # Gene Pos seq: [p1, ..., pn, 0, ..., 0] [btz,g_len]
        # Gene Expr seq: [x1, ..., xn, 0, ..., 0] [btz,g_len]
        # Gene Expr mask: [0, ..., 1, 0, ..., 0] [btz,g_len]

    # ATAC module dict:
        # ATAC Pos seq: [p1, ..., pm, 0, ..., 0] [btz,a_len]
        # ATAC Pos mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
        # ATAC Expr seq: [s1, ..., sm, 0, ..., 0] [btz,a_len]
        # ATAC Expr mask: [0, ..., 1, 0, ..., 0] [btz,a_len]

    def __init__(self,
                 gene_vocab_size: int = 19072,  # 3-19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
                                                # 19069 and 19070 for RNA start and end,
                                                # 19071 and 19072 for ATAC start and end
                 pad_id=0,
                 mask_id=1,
                 cls_id=2,
                 RNAstart_id = 19068,
                 RNAend_id = 19069,
                 ATACstart_id = 19070,
                 ATACend_id = 19071,

                 gene_vocab_initW_path = "/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/data/qwen_gene_embeddings.pt", # w to initialize the gene ID embeddings
                 ebdim_total = 1024,

                 pad_value = 0,  # this is what value of pad token should put in for the masked token
                 mask_value = -1, # this is what value of mask token should put in for the masked token

                 dropout: float = 0.1,

                 prompt = "The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. "
                ):
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

        self.gene_vocab_initW_path = gene_vocab_initW_path
        self.ebdim_total = ebdim_total

        self.padvalue = pad_value
        self.maskvalue = mask_value

        self.dropout = dropout

        # get prompt embeddings
        self.prompt = prompt
        if self.prompt is not None:
            prompt_embedder = PromptEmbedder()
            self.prompt_embeddings = prompt_embedder.encode_text(self.prompt)  #[1,seq_len,embd]

        ##### define sub-modules whose parameters are needed updating
        # Gene and special token id embedding
        self.ID_emb = EmbeddingLNorm(self.gene_vocab_size, self.ebdim_total, self.pad_id)
        # initialize it with Qwen embeddings
        if self.gene_vocab_initW_path is not None:
            print("Initializing gene ID embeddings")
            gene_vocab_initW = torch.load(self.gene_vocab_initW_path)
            weights = self.ID_emb.emb.weight.data
            for gene_id, emb_tensor in gene_vocab_initW.items():
                if gene_id < self.gene_vocab_size:  # make sure gene_id is within range
                    weights[gene_id] = emb_tensor
                else:
                    raise ValueError(f"Gene ID {gene_id} is out of range (n={self.gene_vocab_size})")

        # RNA Expr projection adaptor, used to project one float value to a vector
        self.RNA_Expr_proj = MLP(input_dim=1,
                                  adapter_dim=self.ebdim_total,
                                  output_dim=self.ebdim_total,
                                  dropout_prob=self.dropout)

        # ATAC Expr projection adaptor, used to project one float value to a vector
        self.ATAC_Expr_proj = MLP(input_dim = 1,
                                  adapter_dim = self.ebdim_total,
                                  output_dim = self.ebdim_total,
                                  dropout_prob=self.dropout)

        # LLM
        #self.LLM = LLM_test()
        self.LLM = QWENModel(
            model_path = '/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/qwen_model/qwen',
            freeze = True
        )

        # Prediction heads with LLM output:
        # aPos_head: predict discrete tokens (logits over a_vocab_size)
        self.aPos_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1,dropout_prob=self.dropout)
        # aExpr_head and gExpr_head: predict float values (regression).
        self.aExpr_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1,dropout_prob=self.dropout)
        self.gExpr_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1,dropout_prob=self.dropout)

    def genomic_sinusoidal_encoding(self, positions, emb_dim):
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
        normalized_positions = positions.float() / 1  # now in a smaller range (already scale with 1e6 in dataset)
        normalized_positions = normalized_positions.unsqueeze(-1)  # [batch, seq_len, 1]

        # Create a tensor of dimension indices.
        div_term = torch.exp(
            torch.arange(0, emb_dim, 2, dtype=torch.float32, device=self.device) * -(math.log(10000.0) / emb_dim)
        )
        # div_term shape is [emb_dim/2]. We need to ensure it broadcasts correctly.

        pe = torch.zeros(normalized_positions.size(0), normalized_positions.size(1), emb_dim, device=self.device)
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
        mask = (gID == self.pad_id).unsqueeze(-1) # Expand to shape [btz, seq_len, 1]
        gPos_embeddings = gPos_embeddings.masked_fill(mask, self.padvalue)  # make all values of the dims of padding token to 0

        # Embed gExpr
        # make the mask postion to value -1
        mask = (gExpr_mask == 1)
        gExpr = gExpr.masked_fill(mask,self.maskvalue)  # make all values mask position to -1
        # project it to a vector
        gExpr = gExpr.unsqueeze(-1)
        gExpr_embeddings = self.RNA_Expr_proj(gExpr)
        # make the pad embeddings to value 0
        mask = (gID == self.pad_id).unsqueeze(-1)
        gExpr_embeddings = gExpr_embeddings.masked_fill(mask,self.padvalue)

        # Sum up all embeddings
        RNA_embeddings = gID_embeddings + gPos_embeddings + gExpr_embeddings

        # Add RNA start and end tokens
        RNAstart_token = torch.full((gID.shape[0], 1), self.RNAstart_id, device=self.device)
        RNAstart_token_embeddings = self.ID_emb(RNAstart_token)
        RNAend_token = torch.full((gID.shape[0], 1), self.RNAend_id, device=self.device)
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
        ATACstart_token = torch.full((aPos.shape[0], 1), self.ATACstart_id, device=self.device)
        ATACstart_token_embeddings = self.ID_emb(ATACstart_token)
        ATACend_token = torch.full((aPos.shape[0], 1), self.ATACend_id, device=self.device)
        ATACend_token_embeddings = self.ID_emb(ATACend_token)
        ATAC_embeddings = torch.cat((ATACstart_token_embeddings, ATAC_embeddings, ATACend_token_embeddings), dim=1)

        return ATAC_embeddings

    def forward_beforeLLM(self,module_RNA, module_ATAC):
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

        # Encode prompt
        prompt_embeddings = self.prompt_embeddings.expand(RNA_embeddings.shape[0], -1, -1)
        prompt_embeddings = prompt_embeddings.to(self.device)

        # concate sentence
        All_embeddings = torch.concat([prompt_embeddings,RNA_embeddings, ATAC_embeddings],dim=1)

        return All_embeddings

    def forward_test_output(self,module_RNA, module_ATAC):
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

        # Encode prompt
        prompt_embeddings = self.prompt_embeddings.expand(RNA_embeddings.shape[0], -1, -1)
        prompt_embeddings = prompt_embeddings.to(self.device)

        # concate sentence
        All_embeddings = torch.concat([prompt_embeddings, RNA_embeddings, ATAC_embeddings], dim=1)

        # Pass through decoder
        # devide the RNA and ATAC part
        if self.prompt is not None:
            temp1 = [self.prompt_embeddings.shape[1],1, module_RNA['gExpr'].shape[1], 1, 1, module_ATAC['aExpr'].shape[1], 1]
            _, _, RNA_embeddings, _, _, ATAC_embeddings, _ = torch.split(All_embeddings, temp1, dim=1)
        else:
            temp1 = [1, module_RNA['gExpr'].shape[1], 1, 1,module_ATAC['aExpr'].shape[1], 1]
            _, RNA_embeddings, _, _, ATAC_embeddings, _ = torch.split(All_embeddings, temp1, dim=1)

        # predict with decoder
        out_RNA = self.gExpr_head(RNA_embeddings)
        out_ATAC = self.aExpr_head(ATAC_embeddings)

        return out_RNA, out_ATAC

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

        # Encode prompt
        prompt_embeddings = self.prompt_embeddings.expand(RNA_embeddings.shape[0], -1, -1)
        prompt_embeddings = prompt_embeddings.to(self.device)

        # concate sentence
        All_embeddings = torch.concat([prompt_embeddings, RNA_embeddings, ATAC_embeddings], dim=1)

        # get attention mask to filter out padding during calculation
        gmask_attention = (module_RNA['gID'] != self.pad_id).long().to(self.device)
        amask_attention = (module_ATAC['aPos'] != 0).long().to(self.device)
        promptmask_attention = torch.ones((prompt_embeddings.shape[0],prompt_embeddings.shape[1]),dtype=torch.long).to(self.device)
        mask_attention = torch.concat(tensors =[promptmask_attention,
                                               torch.ones((prompt_embeddings.shape[0],1),dtype=torch.long, device=self.device),
                                               gmask_attention,
                                               torch.ones((prompt_embeddings.shape[0], 1), dtype=torch.long, device=self.device),
                                               torch.ones((prompt_embeddings.shape[0], 1), dtype=torch.long, device=self.device),
                                               amask_attention,
                                               torch.ones((prompt_embeddings.shape[0],1),dtype=torch.long, device=self.device)], dim=1)

        # Pass through LLM
        All_embeddings = self.LLM(All_embeddings,mask_attention).hidden_states[-1]

        # Pass through decoder
        # devide the RNA and ATAC part
        if self.prompt is not None:
            temp1 = [self.prompt_embeddings.shape[1],1, module_RNA['gExpr'].shape[1], 1, 1, module_ATAC['aExpr'].shape[1], 1]
            _, _, RNA_embeddings, _, _, ATAC_embeddings, _ = torch.split(All_embeddings, temp1, dim=1)
        else:
            temp1 = [1, module_RNA['gExpr'].shape[1], 1, 1,module_ATAC['aExpr'].shape[1], 1]
            _, RNA_embeddings, _, _, ATAC_embeddings, _ = torch.split(All_embeddings, temp1, dim=1)

        # predict with decoder
        out_RNA = self.gExpr_head(RNA_embeddings)
        out_ATAC = self.aExpr_head(ATAC_embeddings)


        return out_RNA,out_ATAC
class Multiomics_plus_LoRA(Multiomics_plus):
    # input
    # RNA module dict:
        # Gene ID seq: [g1, ..., gn, pad, ..., pad] [btz,g_len]
        # Gene Pos seq: [p1, ..., pn, 0, ..., 0] [btz,g_len]
        # Gene Expr seq: [x1, ..., xn, 0, ..., 0] [btz,g_len]
        # Gene Expr mask: [0, ..., 1, 0, ..., 0] [btz,g_len]

    # ATAC module dict:
        # ATAC Pos seq: [p1, ..., pm, 0, ..., 0] [btz,a_len]
        # ATAC Pos mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
        # ATAC Expr seq: [s1, ..., sm, 0, ..., 0] [btz,a_len]
        # ATAC Expr mask: [0, ..., 1, 0, ..., 0] [btz,a_len]

    def __init__(self,
                 gene_vocab_size: int = 19072,  # 3-19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
                                                # 19069 and 19070 for RNA start and end,
                                                # 19071 and 19072 for ATAC start and end
                 pad_id=0,
                 mask_id=1,
                 cls_id=2,
                 RNAstart_id = 19068,
                 RNAend_id = 19069,
                 ATACstart_id = 19070,
                 ATACend_id = 19071,

                 gene_vocab_initW_path = "/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/data/qwen_gene_embeddings.pt", # w to initialize the gene ID embeddings
                 ebdim_total = 1024,

                 pad_value = 0,  # this is what value of pad token should put in for the masked token
                 mask_value = -1, # this is what value of mask token should put in for the masked token

                 dropout: float = 0.1,

                 prompt = "The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. ",

                 lora_r = 8,
                 lora_alpha = 16
                ):
        """
        Args:
            emb_dim (int): Common embedding dimension.
            num_heads (int): Number of attention heads.
            g_vocab_size (int): Vocabulary size for gID.
            a_vocab_size (int): Vocabulary size for aPos.
            mask_token_init (float): Initial value for mask token embeddings.
        """
        #####init
        super().__init__(gene_vocab_size=gene_vocab_size,
                 pad_id=pad_id,
                 mask_id=mask_id,
                 cls_id=cls_id,
                 RNAstart_id = RNAstart_id,
                 RNAend_id = RNAend_id,
                 ATACstart_id = ATACstart_id,
                 ATACend_id = ATACend_id,
                 gene_vocab_initW_path = gene_vocab_initW_path,
                 ebdim_total = ebdim_total,
                 pad_value = pad_value,
                 mask_value = mask_value,
                 dropout = dropout,
                 prompt = prompt)

        # set lora config
        self.lora_config = LoraConfig(
            r=lora_r,  # LoRA rank
            lora_alpha=lora_alpha,  # scaling factor
            target_modules=["q_proj", "v_proj"],  # modules to apply LoRA on
            lora_dropout=self.dropout,
            bias="none"
        )

        # set lora to Qwen
        self.LLM = get_peft_model(self.LLM, self.lora_config)

class Multiomics_plus_LoRA_stat(Multiomics_plus_LoRA):
    # input
    # RNA module dict:
        # Gene ID seq: [g1, ..., gn, pad, ..., pad] [btz,g_len]
        # Gene Pos seq: [p1, ..., pn, 0, ..., 0] [btz,g_len]
        # Gene Expr seq: [x1, ..., xn, 0, ..., 0] [btz,g_len]
        # Gene Expr mask: [0, ..., 1, 0, ..., 0] [btz,g_len]

    # ATAC module dict:
        # ATAC Pos seq: [p1, ..., pm, 0, ..., 0] [btz,a_len]
        # ATAC Pos mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
        # ATAC Expr seq: [s1, ..., sm, 0, ..., 0] [btz,a_len]
        # ATAC Expr mask: [0, ..., 1, 0, ..., 0] [btz,a_len]

    def __init__(self,
                 gene_vocab_size: int = 19072,  # 3-19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
                                                # 19069 and 19070 for RNA start and end,
                                                # 19071 and 19072 for ATAC start and end
                 pad_id=0,
                 mask_id=1,
                 cls_id=2,
                 RNAstart_id = 19068,
                 RNAend_id = 19069,
                 ATACstart_id = 19070,
                 ATACend_id = 19071,

                 gene_vocab_initW_path = "/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/data/qwen_gene_embeddings.pt", # w to initialize the gene ID embeddings
                 ebdim_total = 1024,

                 pad_value = 0,  # this is what value of pad token should put in for the masked token
                 mask_value = -1, # this is what value of mask token should put in for the masked token

                 dropout: float = 0.1,

                 prompt = "The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. ",

                 lora_r = 8,
                 lora_alpha = 16,

                 task='ATAC_generation'
                ):
        """
        This is used to add statistic info as well
        """
        #####init
        super().__init__(gene_vocab_size=gene_vocab_size,
                 pad_id=pad_id,
                 mask_id=mask_id,
                 cls_id=cls_id,
                 RNAstart_id = RNAstart_id,
                 RNAend_id = RNAend_id,
                 ATACstart_id = ATACstart_id,
                 ATACend_id = ATACend_id,
                 gene_vocab_initW_path = gene_vocab_initW_path,
                 ebdim_total = ebdim_total,
                 pad_value = pad_value,
                 mask_value = mask_value,
                 dropout = dropout,
                 prompt = prompt,
                 lora_r=lora_r,
                 lora_alpha=lora_alpha
                )

        self.task = task

        # add a new MLP for stat embedding
        self.RNA_Stat_proj = MLP(input_dim=7,
                                 adapter_dim=self.ebdim_total,
                                 output_dim=self.ebdim_total,
                                 dropout_prob=self.dropout)

        if self.task == 'random_NTP':
            self.gExpr_head = MLP(input_dim=self.ebdim_total*2,
                                 adapter_dim=self.ebdim_total*2,
                                 output_dim=1,
                                 dropout_prob=self.dropout)
            self.aExpr_head = MLP(input_dim=self.ebdim_total*2,
                                 adapter_dim=self.ebdim_total*2,
                                 output_dim=1,
                                 dropout_prob=self.dropout)


    def task_embeddings(self, module_RNA, module_ATAC):
        if self.task == 'MLM':
            # Encode each module.
            RNA_embeddings = self.Embed_RNA(module_RNA)
            ATAC_embeddings = self.Embed_ATAC(module_ATAC)

            # define order
            order = 0 # 0: [RNA,ATAC], 1:[ATAC,RNA]
            if order == 0:
                tensor_list = [RNA_embeddings, ATAC_embeddings]
            else:
                tensor_list = [ATAC_embeddings, RNA_embeddings]

            # concate sentence
            All_embeddings_noprompt = torch.concat(tensor_list, dim=1)

        elif self.task == 'random_NTP':
            # this is used to do next token prediction.
            # 'random' means RNA and ATAC is presented with randomly order, i.e., [RNA][ATAC] or [ATAC][RNA]
            aPos = module_ATAC['aPos']
            aExpr_mask = module_ATAC["aExpr_mask"]
            gID = module_RNA['gID']
            gExpr_mask = module_RNA["gExpr_mask"]

            # make all ATAC expr unmasked
            mask = (aPos != 0)
            aExpr_mask = aExpr_mask.masked_fill(mask, 0)
            module_ATAC["aExpr_mask"] = aExpr_mask

            # make all RNA expr unmasked
            mask = (gID != self.pad_id)
            gExpr_mask = gExpr_mask.masked_fill(mask, 0)
            module_RNA["gExpr_mask"] = gExpr_mask

            # Encode each module.
            RNA_embeddings = self.Embed_RNA(module_RNA)
            ATAC_embeddings = self.Embed_ATAC(module_ATAC)

            # random shuffle
            order = random.randint(0, 1) # 0: [RNA,ATAC], 1:[ATAC,RNA]
            if order == 0:
                tensor_list = [RNA_embeddings, ATAC_embeddings]
            else:
                tensor_list = [ATAC_embeddings,RNA_embeddings]

            # concate sentence
            All_embeddings_noprompt = torch.concat(tensor_list, dim=1)

        elif self.task == 'ATAC_generation':
            aPos = module_ATAC['aPos']
            aExpr_mask = module_ATAC["aExpr_mask"]
            gID = module_RNA['gID']
            gExpr_mask = module_RNA["gExpr_mask"]

            # make all ATAC expr masked
            mask = (aPos != 0)
            aExpr_mask = aExpr_mask.masked_fill(mask, 1)
            module_ATAC["aExpr_mask"] = aExpr_mask

            # make all RNA expr unmasked
            mask = (gID != self.pad_id)
            gExpr_mask = gExpr_mask.masked_fill(mask, 0)
            module_RNA["gExpr_mask"] = gExpr_mask

            # Encode each module.
            RNA_embeddings = self.Embed_RNA(module_RNA)
            ATAC_embeddings = self.Embed_ATAC(module_ATAC)

            # define order
            order = 0  # 0: [RNA,ATAC], 1:[ATAC,RNA]
            if order == 0:
                tensor_list = [RNA_embeddings, ATAC_embeddings]
            else:
                tensor_list = [ATAC_embeddings, RNA_embeddings]

            # concate sentence
            All_embeddings_noprompt = torch.concat(tensor_list, dim=1)


        elif self.task == 'RNA_generation':
            aPos = module_ATAC['aPos']
            aExpr_mask = module_ATAC["aExpr_mask"]
            gID = module_RNA['gID']
            gExpr_mask = module_RNA["gExpr_mask"]

            # make all ATAC expr unmasked
            mask = (aPos != 0)
            aExpr_mask = aExpr_mask.masked_fill(mask, 0)
            module_ATAC["aExpr_mask"] = aExpr_mask

            # make all RNA expr masked
            mask = (gID != self.pad_id)
            gExpr_mask = gExpr_mask.masked_fill(mask, 1)
            module_RNA["gExpr_mask"] = gExpr_mask

            # Encode each module.
            RNA_embeddings = self.Embed_RNA(module_RNA)
            ATAC_embeddings = self.Embed_ATAC(module_ATAC)

            # define order !!!ATAC first!!!
            order = 1  # 0: [RNA,ATAC], 1:[ATAC,RNA]
            if order == 0:
                tensor_list = [RNA_embeddings, ATAC_embeddings]
            else:
                tensor_list = [ATAC_embeddings, RNA_embeddings]

            # concate sentence
            All_embeddings_noprompt = torch.concat(tensor_list, dim=1)

        return All_embeddings_noprompt, order

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
        gStat = module_RNA["gStat"]

        # Embed gID
        gID_embeddings = self.ID_emb(gID)

        # Embed gPos with positional encoding
        gPos_embeddings = self.genomic_sinusoidal_encoding(gPos, self.ebdim_total)
        # make the pad embeddings to value 0
        mask = (gID == self.pad_id).unsqueeze(-1) # Expand to shape [btz, seq_len, 1]
        gPos_embeddings = gPos_embeddings.masked_fill(mask, self.padvalue)  # make all values of the dims of padding token to 0

        # Embed gExpr
        # make the mask postion to value -1
        mask = (gExpr_mask == 1)
        gExpr = gExpr.masked_fill(mask,self.maskvalue)  # make all values mask position to -1
        # project it to a vector
        gExpr = gExpr.unsqueeze(-1)
        gExpr_embeddings = self.RNA_Expr_proj(gExpr)
        # make the pad embeddings to value 0
        mask = (gID == self.pad_id).unsqueeze(-1)
        gExpr_embeddings = gExpr_embeddings.masked_fill(mask,self.padvalue)

        # Embed gStat
        # make the mask postion to value -1
        mask = (gExpr_mask == 1).unsqueeze(-1)
        gStat = gStat.masked_fill(mask, self.maskvalue)
        # project it to embeddings
        gStat_embeddings = self.RNA_Stat_proj(gStat)
        # make the pad embeddings to value 0
        pad_mask = (gID == self.pad_id).unsqueeze(-1)
        gStat_embeddings = gStat_embeddings.masked_fill(pad_mask, self.padvalue)

        # Sum up all embeddings
        RNA_embeddings = gID_embeddings + gPos_embeddings + gExpr_embeddings + gStat_embeddings

        # Add RNA start and end tokens
        RNAstart_token = torch.full((gID.shape[0], 1), self.RNAstart_id, device=self.device)
        RNAstart_token_embeddings = self.ID_emb(RNAstart_token)
        RNAend_token = torch.full((gID.shape[0], 1), self.RNAend_id, device=self.device)
        RNAend_token_embeddings = self.ID_emb(RNAend_token)
        RNA_embeddings = torch.cat((RNAstart_token_embeddings, RNA_embeddings, RNAend_token_embeddings), dim=1)

        return RNA_embeddings

    def Embed_RNA_noExpr(self, module_RNA):
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
        gStat = module_RNA["gStat"]

        # Embed gID
        gID_embeddings = self.ID_emb(gID)

        # Embed gPos with positional encoding
        gPos_embeddings = self.genomic_sinusoidal_encoding(gPos, self.ebdim_total)
        # make the pad embeddings to value 0
        mask = (gID == self.pad_id).unsqueeze(-1)  # Expand to shape [btz, seq_len, 1]
        gPos_embeddings = gPos_embeddings.masked_fill(mask,self.padvalue)  # make all values of the dims of padding token to 0

        # Embed gStat
        # make the mask postion to value -1
        mask = (gExpr_mask == 1).unsqueeze(-1)
        gStat = gStat.masked_fill(mask, self.maskvalue)
        # project it to embeddings
        gStat_embeddings = self.RNA_Stat_proj(gStat)
        # make the pad embeddings to value 0
        pad_mask = (gID == self.pad_id).unsqueeze(-1)
        gStat_embeddings = gStat_embeddings.masked_fill(pad_mask, self.padvalue)

        # Sum up all embeddings
        RNA_embeddings = gID_embeddings + gPos_embeddings + gStat_embeddings

        return RNA_embeddings

    def Embed_ATAC_noExpr(self, module_ATAC):
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

        # Sum up all embeddings
        ATAC_embeddings = aPos_embeddings

        return ATAC_embeddings

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

        # task specific embeddings
        All_embeddings_noprompt, order = self.task_embeddings(module_RNA,module_ATAC)

        # Encode prompt
        prompt_embeddings = self.prompt_embeddings.expand(All_embeddings_noprompt.shape[0], -1, -1)
        prompt_embeddings = prompt_embeddings.to(self.device)

        # concate sentence
        All_embeddings = torch.concat([prompt_embeddings, All_embeddings_noprompt], dim=1)

        # get attention mask to filter out padding during calculation
        gmask_attention = (module_RNA['gID'] != self.pad_id).long().to(self.device)
        amask_attention = (module_ATAC['aPos'] != 0).long().to(self.device)
        promptmask_attention = torch.ones((prompt_embeddings.shape[0],prompt_embeddings.shape[1]),dtype=torch.long).to(self.device)
        mask_attention = torch.concat(tensors =[promptmask_attention,
                                               torch.ones((prompt_embeddings.shape[0],1),dtype=torch.long, device=self.device),
                                               gmask_attention,
                                               torch.ones((prompt_embeddings.shape[0], 1), dtype=torch.long, device=self.device),
                                               torch.ones((prompt_embeddings.shape[0], 1), dtype=torch.long, device=self.device),
                                               amask_attention,
                                               torch.ones((prompt_embeddings.shape[0],1),dtype=torch.long, device=self.device)], dim=1)

        # Pass through LLM
        All_embeddings = self.LLM(All_embeddings,mask_attention).hidden_states[-1]

        if self.task == 'random_NTP':
            # devide the RNA and ATAC part
            if order == 0:
                temp1 = [self.prompt_embeddings.shape[1], module_RNA['gExpr'].shape[1]+1, 1,module_ATAC['aExpr'].shape[1]+1, 1]
                _, s_RNA_embeddings, _, s_ATAC_embeddings, _ = torch.split(All_embeddings, temp1, dim=1)
            else:
                temp1 = [self.prompt_embeddings.shape[1], module_ATAC['aExpr'].shape[1]+1, 1,module_RNA['gExpr'].shape[1]+1, 1]
                _, s_ATAC_embeddings, _, s_RNA_embeddings, _ = torch.split(All_embeddings, temp1, dim=1)

            # make the embeddings shift
            RNA_embeddings = s_RNA_embeddings[:, :-1, :] # delete the last token
            ATAC_embeddings = s_ATAC_embeddings[:, :-1, :]  # delete the last token

            # embedding the tokens without expression to incorporate other information
            RNA_embeddings_noExpr = self.Embed_RNA_noExpr(module_RNA)
            ATAC_embeddings_noExpr = self.Embed_ATAC_noExpr(module_ATAC)

            # concatenate
            RNA_embeddings = torch.concat([RNA_embeddings,RNA_embeddings_noExpr], dim = 2)
            ATAC_embeddings = torch.concat([ATAC_embeddings, ATAC_embeddings_noExpr], dim = 2)

        else:
            # devide the RNA and ATAC part
            if order == 0:
                temp1 = [self.prompt_embeddings.shape[1], 1, module_RNA['gExpr'].shape[1], 1, 1, module_ATAC['aExpr'].shape[1],1]
                _, _, RNA_embeddings, _, _, ATAC_embeddings, _ = torch.split(All_embeddings, temp1, dim=1)
            else:
                temp1 = [self.prompt_embeddings.shape[1], 1, module_ATAC['aExpr'].shape[1], 1, 1, module_RNA['gExpr'].shape[1], 1]
                _, _, ATAC_embeddings, _, _, RNA_embeddings, _ = torch.split(All_embeddings, temp1, dim=1)

        # predict with decoder
        out_RNA = self.gExpr_head(RNA_embeddings)
        out_ATAC = self.aExpr_head(ATAC_embeddings)

        return out_RNA,out_ATAC,order


# Example usage:
if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from tool.demo_generator import demo_input

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # generate demo minibatch
    generator = demo_input( n_batch=4,
                            gene_len=None,
                            atac_len=None,
                            gene_total_len=10,
                            atac_total_len=15,
                            gExpr_mask_frac=0.15,
                            aPos_mask_frac=0,
                            aExpr_mask_frac=0.15,
                            addPadding=True,
                            pad_id=0,
                            max_gene_num=1000,
                            max_expr=10,
                            max_pos=1000000,
                            addStat=True,
                            stat_dim=7,
                            max_stat=10,
                            device = device)
    RNA_module, ATAC_module = generator.generate_data()

    # init moodel
    # Mymodel=Multiomics_plus(gene_vocab_size= 19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
    #                          pad_id=0,
    #                          mask_id=1,
    #                          cls_id=2,
    #                          RNAstart_id=19068,
    #                          RNAend_id=19069,
    #                          ATACstart_id=19070,
    #                          ATACend_id=19071,
    #                          gene_vocab_initW_path = "/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/data/qwen_gene_embeddings.pt",
    #                          ebdim_total=896,
    #                          pad_value=0,
    #                          mask_value=-1,
    #                          dropout = 0.1,
    #                          prompt="The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. "
    #                         )

    # Mymodel=Multiomics_plus_LoRA(gene_vocab_size= 19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
    #                              pad_id=0,
    #                              mask_id=1,
    #                              cls_id=2,
    #                              RNAstart_id=19068,
    #                              RNAend_id=19069,
    #                              ATACstart_id=19070,
    #                              ATACend_id=19071,
    #                              gene_vocab_initW_path = "/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/data/qwen_gene_embeddings.pt",
    #                              ebdim_total=896,
    #                              pad_value=0,
    #                              mask_value=-1,
    #                              dropout = 0.1,
    #                              prompt="The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. ",
    #                              lora_r = 8,
    #                              lora_alpha = 16
    #                             )

    # Mymodel = Multiomics_plus_LoRA(gene_vocab_size=19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
    #                                 pad_id=0,
    #                                 mask_id=1,
    #                                 cls_id=2,
    #                                 RNAstart_id=19068,
    #                                 RNAend_id=19069,
    #                                 ATACstart_id=19070,
    #                                 ATACend_id=19071,
    #                                 gene_vocab_initW_path = "/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/data/qwen_gene_embeddings.pt",
    #                                 ebdim_total=896,
    #                                 pad_value=0,
    #                                 mask_value=-1,
    #                                 dropout = 0.1,
    #                                 prompt="The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. ",
    #                                 lora_r = 8,
    #                                 lora_alpha = 16
    #                                )
    Mymodel = Multiomics_plus_LoRA_stat(gene_vocab_size=19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
                                           pad_id=0,
                                           mask_id=1,
                                           cls_id=2,
                                           RNAstart_id=19068,
                                           RNAend_id=19069,
                                           ATACstart_id=19070,
                                           ATACend_id=19071,
                                           gene_vocab_initW_path="/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/data/qwen_gene_embeddings.pt",
                                           ebdim_total=896,
                                           pad_value=0,
                                           mask_value=-1,
                                           dropout=0.1,
                                           prompt="The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. ",
                                           lora_r=8,
                                           lora_alpha=16,
                                           task = 'random_NTP'
                                           )


    # set device
    Mymodel.to(device)
    Mymodel.device = device

    # test terms of lora exist
    for name, param in Mymodel.LLM.named_parameters():
        if "q_proj" in name or "v_proj" in name:
            print(name)
    for name, param in Mymodel.LLM.named_parameters():
        print(f"{name}: {'Trainable' if param.requires_grad else 'Frozen'}")

    # test forward
    out_RNA, out_ATAC, order = Mymodel(RNA_module, ATAC_module)

    # # test forward before LLM
    # LMM_input = Mymodel.forward_beforeLLM(RNA_module, ATAC_module)

    # # test forward before LLM
    # out_RNA, out_ATAC = Mymodel.forward_test_output(RNA_module, ATAC_module)

    # test loss
    MSE_loss = nn.MSELoss()

    if Mymodel.task == 'random_NTP':
        RNApad_mask = (RNA_module['gID'] != Mymodel.pad_id).unsqueeze(-1)
        ATACpad_mask = (ATAC_module['aPos'] != 0).unsqueeze(-1)
        if order == 0:
            RNApad_mask[:,:3, :] = False # do not calculate the loss with first 1024 tokens
        else:
            ATACpad_mask[:, :3, :] = False  # do not predict the first 1024 tokens
        temp_mask = torch.concat([RNApad_mask, ATACpad_mask], dim=1)
        predict = torch.concat([out_RNA, out_ATAC], dim=1)[temp_mask]
        target = torch.concat([RNA_module['gExpr'], ATAC_module['aExpr']], dim=1).unsqueeze(-1)[temp_mask]
        loss = MSE_loss(predict, target)
    else:
        temp_res = torch.concat([out_RNA, out_ATAC], dim=1)
        temp_mask = torch.concat([RNA_module['gExpr_mask'], ATAC_module['aExpr_mask']], dim=1).bool()
        predict = temp_res[temp_mask]
        target = torch.concat([RNA_module['gExpr'], ATAC_module['aExpr']], dim=1).unsqueeze(-1)[temp_mask]
        loss = MSE_loss(predict, target)

    temp_loss = MSE_loss(predict, target)



