#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/03/21 15:13
# @Author  : Kieran Wang
# @File    : model_tmp.py
# @Software: PyCharm


# class Multiomics_sep(nn.Module):
#     # input
#     # Gene ID seq: [g1, ..., gn, pad, ..., pad] [btz,g_len]
#     # Gene Pos seq: [p1, ..., pn, 0, ..., 0] [btz,g_len]
#     # Gene Expr seq: [x1, ..., xn, 0, ..., 0] [btz,g_len]
#     # Gene Expr mask: [0, ..., 1, 0, ..., 0] [btz,g_len]
#
#     # ATAC Pos seq: [p1, ..., pm, 0, ..., 0] [btz,a_len]
#     # ATAC Pos mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
#     # ATAC Expr seq: [s1, ..., sm, 0, ..., 0] [btz,a_len]
#     # ATAC Expr mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
#
#     def __init__(self,
#                  gene_vocab_size: int = 19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
#                                                 # 19069 and 19070 for RNA start and end,
#                                                 # 19071 and 19072 for ATAC start and end
#                  pad_id=0,
#                  mask_id=1,
#                  cls_id=2,
#                  RNA_start = 19069,
#                  RNA_end = 19070,
#                  ATAC_start = 19071,
#                  ATAC_end = 19072,
#
#
#                  ebdim_gID = 512,
#                  ebdim_gPos = 256,
#                  ebdim_gExpr = 256,
#                  ebdim_aPos = 256,
#                  ebdim_aExpr = 256,
#
#                  ebd_padvalue = 0,  # this is what value of every dim of embd should put in for the masked token
#                  ebd_maskvalue = -1, # this is what value of every dim of embd should put in for the masked token
#
#
#                  nhead: int = 16,
#                  num_layers: int = 2,
#                  dropout: float = 0.1,
#                  token_embed_method='forward_embedding_multiply',
#                  mask_token_init=0.0):
#         """
#         Args:
#             emb_dim (int): Common embedding dimension.
#             num_heads (int): Number of attention heads.
#             g_vocab_size (int): Vocabulary size for gID.
#             a_vocab_size (int): Vocabulary size for aPos.
#             mask_token_init (float): Initial value for mask token embeddings.
#         """
#         #####init
#         super(MultiModuleCompletionModel, self).__init__()
#         self.gene_vocab_size = gene_vocab_size
#         self.pad_id = pad_id
#         self.mask_id = mask_id
#         self.cls_id = cls_id
#
#         self.ebdim_gID = ebdim_gID
#         self.ebdim_gPos = ebdim_gPos
#         self.ebdim_gExpr = ebdim_gExpr
#         self.ebdim_aPos = ebdim_aPos
#         self.ebdim_aExpr = ebdim_aExpr
#         self.ebdim_total = ebdim_gID + ebdim_gPos + ebdim_gExpr
#
#         self.ebd_padvalue = ebd_padvalue
#         self.ebd_maskvalue = ebd_maskvalue
#
#         self.nhead = nhead
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.token_embed_method = token_embed_method
#         self.mask_token_init = mask_token_init
#
#         ##### define sub-modules whose parameters are needed updating
#         # Gene id embedding
#         self.gID_emb = EmbeddingLNorm(self.gene_vocab_size, self.ebdim_gID, self.pad_id)
#
#         # ATAC token projection adaptor, used to project it as the same dim with RNA tokens
#         self.ATAC_proj = MLP(input_dim = self.ebdim_aPos + self.ebdim_aExpr,
#                                  adapter_dim = self.ebdim_total,
#                                  output_dim = self.ebdim_total)
#
#         # Cross-attention modules (using batch_first=True)
#         # For module1: Query comes from RNA; key/value come from ATAC.
#         self.attn_QRNA = CrossAttentionFFN(embd=self.ebdim_total, num_heads=self.nhead, ffn_hidden_dim=None, dropout=0.1, batch_first=True)
#         # For module2: Query comes from ATAC; key/value come from RNA.
#         self.attn_QATAC = CrossAttentionFFN(embd=self.ebdim_total, num_heads=self.nhead, ffn_hidden_dim=None, dropout=0.1, batch_first=True)
#
#         # Adaptor to LLM
#         self.LLMadaptor = LLMadapter_test()
#
#         # Prediction heads with LLM output:
#         # aPos_head: predict discrete tokens (logits over a_vocab_size)
#         self.aPos_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1)
#         # aExpr_head and gExpr_head: predict float values (regression).
#         self.aExpr_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1)
#         self.gExpr_head = MLP(input_dim = self.ebdim_total,adapter_dim = self.ebdim_total, output_dim = 1)
#
#     @staticmethod
#     def genomic_sinusoidal_encoding(positions, emb_dim):
#         """
#         Compute sinusoidal positional encodings for genomic positions.
#
#         Args:
#             positions (Tensor): Tensor of shape [batch, seq_len] containing genomic positions as integers.
#             emb_dim (int): Dimension of the positional encoding.
#             scale (float): Factor to normalize the positions (e.g., 1e6).
#
#         Returns:
#             Tensor: Positional encodings of shape [batch, seq_len, emb_dim].
#         """
#         # Normalize positions to reduce the scale.
#         normalized_positions = positions.float()  # now in a smaller range
#         normalized_positions = normalized_positions.unsqueeze(-1)  # [batch, seq_len, 1]
#
#         # Create a tensor of dimension indices.
#         div_term = torch.exp(torch.arange(0, emb_dim, 2, dtype=torch.float32) * -(math.log(10000.0) / emb_dim))
#         # div_term shape is [emb_dim/2]. We need to ensure it broadcasts correctly.
#
#         pe = torch.zeros(normalized_positions.size(0), normalized_positions.size(1), emb_dim)
#         pe[:, :, 0::2] = torch.sin(normalized_positions * div_term)
#         pe[:, :, 1::2] = torch.cos(normalized_positions * div_term)
#         return pe
#
#
#     # Get the RNAseq embd by concatenate gID, gPos, and gExpr
#     def Embed_RNA(self, module_RNA):
#         """
#         Encodes RNA features.
#         Expects RNA to be a dict with keys:
#             # gID: [g1, ..., gn, pad, ..., pad] [btz,g_len]
#             # gPos: [p1, ..., pn, 0, ..., 0] [btz,g_len]
#             # gExpr: [x1, ..., xn, 0, ..., 0] [btz,g_len]
#             # gExpr_mask: [0, ..., 1, 0, ..., 0] [btz,g_len]
#         Returns:
#           Tensor of shape [batch, g_len, emb_dim].
#         """
#         gID = module_RNA["gID"]
#         gPos = module_RNA["gPos"]
#         gExpr = module_RNA["gExpr"]
#         gExpr_mask = module_RNA["gExpr_mask"]
#
#         # Embed gID
#         gID_embeddings = self.gID_emb(gID)
#
#         # Embed gPos with positional encoding
#         gPos_embeddings = self.genomic_sinusoidal_encoding(gPos, self.ebdim_gPos)
#         mask = (gPos == 0).unsqueeze(-1)  # Expand to shape [btz, seq_len, 1]
#         gPos_embeddings = gPos_embeddings.masked_fill(mask, self.ebd_padvalue)  # make all values of the dims of padding token to 0
#
#         # Embed gExpr
#         gExpr_embeddings = gExpr.unsqueeze(-1).expand(-1, -1, self.ebdim_gExpr)
#         mask = (gExpr_mask == 1).unsqueeze(-1)  # Expand to shape [btz, seq_len, 1]
#         gExpr_embeddings = gExpr_embeddings.masked_fill(mask,self.ebd_maskvalue)  # make all values of the dims of mask position to 0
#
#         # Concatenate all embeddings
#         RNA_embeddings = torch.cat([gID_embeddings, gPos_embeddings, gExpr_embeddings],dim=2)
#
#         return RNA_embeddings
#
#         # Get the RNAseq embd by concatenate gID, gPos, and gExpr
#
#     def Embed_ATAC(self, module_ATAC):
#         """
#         Encodes ATAC features.
#         Expects ATAC to be a dict with keys:
#             # aPos: [p1, ..., pn, 0, ..., 0] [btz,a_len]
#             # aPos_mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
#             # aExpr: [x1, ..., xn, 0, ..., 0] [btz,a_len]
#             # aExpr_mask: [0, ..., 1, 0, ..., 0] [btz,a_len]
#         Returns:
#           Tensor of shape [batch, a_len, emb_dim].
#         """
#         aPos = module_ATAC["aPos"]
#         aPos_mask = module_ATAC["aPos_mask"]
#         aExpr = module_ATAC["aExpr"]
#         aExpr_mask = module_ATAC["aExpr_mask"]
#
#         # Embed aPos with positional encoding
#         aPos_embeddings = self.genomic_sinusoidal_encoding(aPos, self.ebdim_aPos)
#         mask = (aPos == 0).unsqueeze(-1)  # Expand to shape [btz, seq_len, 1]
#         aPos_embeddings = aPos_embeddings.masked_fill(mask,self.ebd_padvalue)  # make all values of the dims of padding token to 0
#         mask = (aPos_mask == 1).unsqueeze(-1)  # Expand to shape [btz, seq_len, 1]
#         aPos_embeddings = aPos_embeddings.masked_fill(mask,self.ebd_maskvalue)  # make all values of the dims of mask position to 0
#
#         # Embed aExpr
#         aExpr_embeddings = aExpr.unsqueeze(-1).expand(-1, -1, self.ebdim_aExpr)
#         mask = (aExpr_mask == 1).unsqueeze(-1)  # Expand to shape [btz, seq_len, 1]
#         aExpr_embeddings = aExpr_embeddings.masked_fill(mask,self.ebd_maskvalue)  # make all values of the dims of mask position to 0
#
#         # Concatenate all embeddings
#         ATAC_embeddings = torch.cat([aPos_embeddings, aExpr_embeddings], dim=2)
#
#         # Project to total embedding dims
#         ATAC_embeddings = self.ATAC_proj(ATAC_embeddings)
#
#         return ATAC_embeddings
#
#
#     def task_determine(self,module_RNA, module_ATAC):
#
#         # get data
#         gExpr_mask = module_RNA["gExpr_mask"]
#         aPos = module_ATAC["aPos"]
#         aPos_mask = module_ATAC["aPos_mask"]
#         aExpr_mask = module_ATAC["aExpr_mask"]
#
#         # check whether contains all 0
#         (gExpr_mask == 0).all().item()
#
#         # 1. given unmasked gID and gExpr, all-masked aPos and aExpr, predict the actual aPos;
#         # 2. given unmasked gID and gExpr, partially-masked aPos and all-masked aExpr, predict the actual aPos left;
#         # 3. given unmasked gID, gExpr, and aPos, and all-masked aExpr, predict the actual aExpr;
#         # 4. given unmasked gID, gExpr, aPos, and partially-masked aExpr, predict the actual aExpr left;
#         # 5. given unmasked aPos, aExpr, gID, all masked gExpr, predict the actual gExpr;
#         # 6. given unmasked aPos, aExpr, gID, partially masked gExpr, predict the actual gExpr left.
#
#
#
#
#     def forward(self,module_RNA, module_ATAC):
#         """
#         Args:
#           module1 (dict): contains keys "gID", "gExpr" and optionally "gExpr_mask" of shape [batch, g_len]
#           module2 (dict): contains keys "aPos", "aExpr" and optionally "aPos_mask", "aExpr_mask" of shape [batch, a_len]
#           task (int): task id (1 to 6) indicating which prediction to make.
#
#         Returns:
#           For tasks 1-4, predictions are made on module2 outputs (for aPos or aExpr).
#           For tasks 5-6, predictions are made on module1 outputs (for gExpr).
#           The returned tensor(s) only include predictions for positions that were masked.
#         """
#
#         # Encode each module.
#         RNA_embeddings = self.Embed_RNA(module_RNA)
#         ATAC_embeddings = self.Embed_ATAC(module_ATAC)
#
#         # check task type
#
#
#
#
#         # Apply cross-attention:
#         # For module1: query from module2, key/value from module1.
#         module1_out, _ = self.attn_module1(query=module2_emb, key=module1_emb, value=module1_emb)
#         # For module2: query from module1, key/value from module2.
#         module2_out, _ = self.attn_module2(query=module1_emb, key=module2_emb, value=module2_emb)
#
#         # Based on the task, select the output and apply the proper prediction head.
#         if task in [1, 2]:
#             # Predict aPos (discrete tokens) using module2's output.
#             # Use the provided aPos_mask to select positions to predict.
#             if "aPos_mask" not in module2:
#                 raise ValueError("aPos_mask must be provided for task {}".format(task))
#             mask = module2["aPos_mask"]  # shape: [batch, a_len] (bool)
#             # Gather outputs for masked positions.
#             masked_out = module2_out[mask]
#             pred_logits = self.aPos_head(masked_out)  # [n_masked, a_vocab_size]
#             return pred_logits
#
#         elif task in [3, 4]:
#             # Predict aExpr (float regression) using module2's output.
#             if "aExpr_mask" not in module2:
#                 raise ValueError("aExpr_mask must be provided for task {}".format(task))
#             mask = module2["aExpr_mask"]  # shape: [batch, a_len]
#             masked_out = module2_out[mask]
#             pred_values = self.aExpr_head(masked_out)  # [n_masked, 1]
#             # Squeeze last dim if desired.
#             return pred_values.squeeze(-1)
#
#         elif task in [5, 6]:
#             # Predict gExpr (float regression) using module1's output.
#             if "gExpr_mask" not in module1:
#                 raise ValueError("gExpr_mask must be provided for task {}".format(task))
#             mask = module1["gExpr_mask"]  # shape: [batch, g_len]
#             masked_out = module1_out[mask]
#             pred_values = self.gExpr_head(masked_out)  # [n_masked, 1]
#             return pred_values.squeeze(-1)
#
#         else:
#             raise ValueError("Invalid task id. Must be one of 1,2,3,4,5,6.")
#
#
# # Example usage:
# if __name__ == '__main__':
#     # Hyperparameters
#     emb_dim = 64
#     num_heads = 8
#     g_vocab_size = 1000
#     a_vocab_size = 500
#
#     model = MultiModuleCompletionModel(emb_dim, num_heads, g_vocab_size, a_vocab_size)
#
#     # Dummy batch parameters.
#     batch_size = 32
#     g_len = 10
#     a_len = 15
#
#     # Module1 inputs (for gID and gExpr); assume all unmasked except when predicting gExpr.
#     module1 = {
#         "gID": torch.randint(0, g_vocab_size, (batch_size, g_len)),
#         "gExpr": torch.rand(batch_size, g_len),
#         # For tasks 5 and 6, you would set some positions as masked.
#         "gExpr_mask": torch.zeros(batch_size, g_len, dtype=torch.bool)
#     }
#
#     # Module2 inputs (for aPos and aExpr); assume:
#     # For tasks 1 and 2, aPos_mask is all True (or partially True for task 2), and aExpr_mask is all True.
#     # For tasks 3 and 4, aExpr_mask is True for positions to predict.
#     module2 = {
#         "aPos": torch.randint(0, a_vocab_size, (batch_size, a_len)),
#         "aExpr": torch.rand(batch_size, a_len),
#         "aPos_mask": torch.ones(batch_size, a_len, dtype=torch.bool),  # for example, all masked for task1
#         "aExpr_mask": torch.ones(batch_size, a_len, dtype=torch.bool)  # all masked for task1
#     }
#
#     # Example: Task 1 prediction (predict aPos)
#     pred_aPos_logits = model(module1, module2, task=1)
#     print("Task 1, predicted aPos logits shape:", pred_aPos_logits.shape)
#
#     # For tasks 3 or 4 (predict aExpr), you might set aExpr_mask to only partially mask positions.
#     # For tasks 5 or 6 (predict gExpr), adjust module1["gExpr_mask"] accordingly.
