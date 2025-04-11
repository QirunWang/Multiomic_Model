#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/04/02 11:11
# @Author  : Kieran Wang
# @File    : test_pretrain.py
# @Software: PyCharm

##############
# try RNA and ATAC mutual-generation with random_NTP
###################
import os
import sys
sys.path.append("./")

from tool.ConfigHandler import ConfigHandler
from task.pretrain import Pretrain
import torch.distributed as dist
import torch

#
# #############################
# if not os.path.exists("./config/test_20250410/"):
#     os.mkdir("./config/test_20250410/")
#
# # initialize dataset and export
# Myhandler = ConfigHandler()
#
# # config for Dataset
# # config for Dataset
# config_dict = {
#                  "pad_id":0,
#                  "pad_Pos_value":0,
#                  "pad_Expr_value":0,
#
#                  "gene_total_len":4096,
#                  # length for gene-related tensor (gID, gPos, gExpr, gExpr_mask) with pading or crop
#                  "atac_total_len":4096,
#                  # length for atac-related tensor (aPos, aPos_mask, aExpr, aExpr_mask) with pading or crop
#
#                  "gExpr_mask_frac":0,  # mask fraction of gExpr
#                  "aPos_mask_frac":0,  # mask fraction of aPos
#                  "aExpr_mask_frac":0,  # mask fraction of aExpr
#
#                  "pos_scaling_factor":1e6,
#
#                  "gene_vocab_path":"./references/gene_vocab_single_Vqiuping.json",
#
#                  "Database_dir" : "/home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_MultiomicsData_1M/clean2/"
# }
# Myhandler.dict_to_json(config_dict=config_dict,
#                        file_name='./test_20250410/Config_Dataset_DDP.json')
#
#
# # config for Dataloader
# config_dict = {
#          "batch_size":2,
#          "num_workers":0,
#          "shuffle":True
# }
# Myhandler.dict_to_json(config_dict=config_dict,
#                        file_name='./test_20250410/Config_Dataloader_DDP.json')
# # !!!note: please add Dataset to key 'dataset' and 'drawable_indices' after load this json when using!!!
#
#
# # config for Model
# config_dict = {
#                  "gene_vocab_size": 19072,  # 19065 genes plus 0 for pad, and 1 for mask, 2 for cls，
#                  "pad_id":0,
#                  "mask_id":1,
#                  "cls_id":2,
#                  "RNAstart_id":19068,
#                  "RNAend_id":19069,
#                  "ATACstart_id":19070,
#                  "ATACend_id":19071,
#                  "gene_vocab_initW_path":"/home/share/huadjyin/home/lishaoshuai/zhaoyinuo/data/qwen_gene_embeddings.pt",
#                  "ebdim_total":896,
#                  "pad_value":0,
#                  "mask_value":-1,
#                  "dropout" : 0.1,
#                  #"prompt":"The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the masked values of gene and peak expression. ",
#                  "prompt":"The dataset contains paired scRNAseq and scATACseq data for a single cell obtained using 10X Multiome. Each token represents information about either a gene or an ATAC peak. For gene tokens, each embedding encodes the gene ID, its expression level, and its genomic position. For ATAC peak tokens, each embedding encodes the peak’s expression level and its genomic location. Both gene tokens and peak tokens are framed by special tokens at the beginning and end to mark their respective boundaries. These tokens are then concatenated together, with a prompt embedding token prepended to the sequence. The model’s task is to predict the all values of gene and peak expression. ",
#                  "lora_r" : 16,
#                  "lora_alpha" : 16,
#                  "task": 'random_NTP'
# }
# Myhandler.dict_to_json(config_dict=config_dict,
#                        file_name='./test_20250410/Config_Model_DDP.json')
#


# # start pretrain
temp1 = Pretrain(device=None,                       # the device, either cpu or cuda.
                                                    #     If None, detect if cuda is available
                 optimizer='AdamW',                  # optimizer
                 lr = 1e-4,                         # the learning rate of the optimizer
                 weight_decay = 0.01,                # the weight_decay rate of the optimizer

                 scheduler = 'StepLR',
                 stepSche_steps = 1000,
                 stepSche_gamma = 0.9,

                 loss_func='MSE',     # the loss function

                 num_epoch=20,                      # epoch number

                 Dataset_config_file='./test_20250410/Config_Dataset_DDP.json',          # Dataset config file in ./config/
                 DataLoader_config_file='./test_20250410/Config_Dataloader_DDP.json',       # DataLoader config file in ./config/
                 Model_config_file='./test_20250410/Config_Model_DDP.json',            # Model config file in ./config/

                 Dataset_use='GetData_LLM_stat',
                 DataLoader_use='MultiomicLoader_stat',
                 Model_use='Multiomics_plus_LoRA_stat',

                 drawableidx_train_path = "./references/drawableidx_train_20250403.npy", # the idx of fraction of data reserved for train
                 drawableidx_valid_path = "./references/drawableidx_valid_20250403.npy", # the idx of fraction of data reserved for valid

                 parallel = 'DDP',    # Model parallel method. 'DDP', 'deepspeed', 'None'

                 checkpoint_dir='./model_checkpoint/pretrain_20250410',
                 checkpoint_step=1000,

                 logger_proj_name='pretrain_20250410_DDP',
                 test_logger_path="./tmp/pretrain_20250410_DDP/logs/",

                 checkpoint_load_path = None,  # if none, start the model from scratch
                 # checkpoint_load_path = None,
                 model_param_load = True,
                 optimizer_param_load = False,
                 scheduler_param_load = False,

                 grad_clip = True

                 )

temp1.train()


####3 node (4 gpu each), one step 4.5s, total 26666 step, 33h, 1.38 days






