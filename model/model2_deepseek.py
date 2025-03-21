#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/03/18 10:22
# @Author  : Kieran Wang
# @File    : model2_deepseek.py
# @Software: PyCharm


import gc

import numpy as np
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
import torch.optim as optim
import math



class EmbeddingLNorm(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_id=None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.norm = nn.LayerNorm(embedding_dim)

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
        div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / emb_dim))
        # div_term shape is [emb_dim/2]. We need to ensure it broadcasts correctly.

        pe = torch.zeros(normalized_positions.size(0), normalized_positions.size(1), emb_dim)
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
        RNAstart_token = torch.full((gID.shape[0], 1), self.RNAstart_id)
        RNAstart_token_embeddings = self.ID_emb(RNAstart_token)
        RNAend_token = torch.full((gID.shape[0], 1), self.RNAend_id)
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
        ATACstart_token = torch.full((aPos.shape[0], 1), self.ATACstart_id)
        ATACstart_token_embeddings = self.ID_emb(ATACstart_token)
        ATACend_token = torch.full((aPos.shape[0], 1), self.ATACend_id)
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



# Example usage:
if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from tool.demo_generator import demo_input

    # generate demo minibatch
    generator = demo_input( n_batch=4,
                            gene_len=5,
                            atac_len=10,
                            gene_total_len=10,
                            atac_total_len=15,
                            gExpr_mask_frac=0.15,
                            aPos_mask_frac=0,
                            aExpr_mask_frac=0.15,
                            max_gene_num=1000,
                            max_expr=10,
                            max_pos=1000000)
    RNA_module, ATAC_module = generator.generate_data()

    # init moodel
    Mymodel=Multiomics_plus(gene_vocab_size= 19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
                             pad_id=0,
                             mask_id=1,
                             cls_id=2,
                             RNAstart_id=19068,
                             RNAend_id=19069,
                             ATACstart_id=19070,
                             ATACend_id=19071,
                             ebdim_total=1024,
                             pad_value=0,
                             mask_value=-1,
                             nhead = 16,
                             num_layers = 2,
                             dropout = 0.1)

    # test forward
    out_RNA, out_ATAC = Mymodel(RNA_module, ATAC_module)






