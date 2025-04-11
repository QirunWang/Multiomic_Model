#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/04/01 13:23
# @Author  : Kieran Wang
# @File    : pretrain.py
# @Software: PyCharm
import sys
sys.path.append("./")
import os
import time
import wandb
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
from deepspeed.monitor.monitor import MonitorMaster,WandbMonitor
from deepspeed.runtime.config import DeepSpeedConfig
from torch.cuda.amp import autocast, GradScaler
from pytorch_memlab import MemReporter



import logging

import torch
import torch.optim as optim
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import random_split

import random
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from tool.ConfigHandler import ConfigHandler
from dataset.GetData import GetData_LLM,GetData_LLM_stat
from dataset.LoadData import MultiomicLoader,MultiomicLoader_stat
from model.model3 import Multiomics_plus,Multiomics_plus_LoRA_stat

from tool.zclip_PJS import ZClip

class Pretrain(object):
    def __init__(self,
                 device=None,                       # the device, either cpu or cuda.
                                                    #     If None, detect if cuda is available
                 optimizer='AdamW',                  # optimizer
                 lr = 1e-4,                         # the learning rate of the optimizer
                 weight_decay = 0.01,                # the weight_decay rate of the optimizer

                 scheduler = 'StepLR',
                 stepSche_steps = 1000,
                 stepSche_gamma = 0.9,

                 loss_func='MSE',     # the loss function

                 num_epoch=20,                      # epoch number

                 Dataset_config_file='Config_ScDatasetLmdbs_QP_Siamese_20250211.json',          # Dataset config file in ./config/
                 DataLoader_config_file='Config_Prob_BatchLoader_20250211.json',       # DataLoader config file in ./config/
                 Model_config_file='Config_SiameseTransformer_20250211.json',            # Model config file in ./config/

                 Dataset_use = 'GetData_LLM',
                 DataLoader_use = 'MultiomicLoader',
                 Model_use = 'Multiomics_plus',

                 drawableidx_train_path="./references/drawableidx_train_20250403.npy",# the idx of fraction of data reserved for train
                 drawableidx_valid_path="./references/drawableidx_valid_20250403.npy", # the idx of fraction of data reserved for valid

                 parallel = 'DDP',    # Model parallel method. 'DDP', 'deepspeed', 'None'

                 checkpoint_dir = './model_checkpoint/pretrain_20250217_deepspeed',
                 checkpoint_step = None,

                 logger_proj_name = None,
                 test_logger_path = None, # used in logging out the record of each gpu

                 checkpoint_load_path = None,  # if none, start the model from scratch
                 model_param_load = True,
                 optimizer_param_load = True,
                 scheduler_param_load = True,

                 grad_clip = True # use zclip to clip grad to aviod spikes in loss

                 ):


        print("\n\n\n#--------------------------------------Pretraining--------------------------------------#")

        # set device
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        print('[-----pretrain.py]device:', self.device)

        # load parameters
        print("[-----pretrain.py]Paramters: ")
        config_handler = ConfigHandler()  # ConfigHandler
        self.Dataset_config = config_handler.json_to_dict(Dataset_config_file)
        print(self.Dataset_config)
        self.DataLoader_config = config_handler.json_to_dict(DataLoader_config_file)
        print(self.DataLoader_config)
        self.Model_config = config_handler.json_to_dict(Model_config_file)
        print(self.Model_config)

        # instantiations of Dataset
        if Dataset_use == 'GetData_LLM':
            self.Dataset = GetData_LLM(**self.Dataset_config)
        elif Dataset_use == 'GetData_LLM_stat':
            print("[-----pretrain.py]Enabling stat info")
            self.Dataset = GetData_LLM_stat(**self.Dataset_config)

        #get drawable idx
        self.drawableidx_train_path = drawableidx_train_path
        self.drawableidx_valid_path = drawableidx_valid_path
        self.drawableidx_train = list(np.load(self.drawableidx_train_path))
        self.drawableidx_valid = list(np.load(self.drawableidx_valid_path))
        print(f'[-----pretrain.py]train data size (drawableidx_train): {len(self.drawableidx_train)})')

        # instantiations of model
        if Model_use == 'Multiomics_plus':
            self.model = Multiomics_plus(**self.Model_config)
        elif Model_use == 'Multiomics_plus_LoRA_stat':
            print("[-----pretrain.py]Enabling lora mode")
            self.model = Multiomics_plus_LoRA_stat(**self.Model_config)
            print(f"[-----pretrain.py]Training task: {self.model.task}")


        #load checkpoint params to the model
        self.checkpoint_load_path = checkpoint_load_path
        self.model_param_load = model_param_load
        self.optimizer_param_load = optimizer_param_load
        self.scheduler_param_load = scheduler_param_load
        if self.checkpoint_load_path is not None:
            print(f"[-----pretrain.py]load previous trained weights from {self.checkpoint_load_path}")
            checkpoint = torch.load(self.checkpoint_load_path)
            if self.model_param_load:
                self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # set optimizer's para
        self.lr = lr
        self.weight_decay = weight_decay

        if optimizer == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.checkpoint_load_path is not None and self.optimizer_param_load:
            print(f"[-----pretrain.py]load previous status of optimizer from {self.checkpoint_load_path}")
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # set scheduler's para
        self.stepSche_steps = stepSche_steps
        self.stepSche_gamma = stepSche_gamma

        if scheduler == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer)
        elif scheduler == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=self.stepSche_steps,
                                                       gamma=self.stepSche_gamma)
        print(f"[-----pretrain.py]Scheduler {'enabled' if hasattr(self, 'scheduler') else 'disabled'}")
        if self.checkpoint_load_path is not None and self.scheduler_param_load:
            print(f"[-----pretrain.py]load previous status of scheduler from {self.checkpoint_load_path}")
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # set scaler
        self.scaler = GradScaler(enabled=True)
        print(f"[-----pretrain.py]Scaler {'enabled' if hasattr(self, 'scaler') else 'disabled'}")

        # define loss
        if loss_func=='MSE':
            self.loss_func = nn.MSELoss()

        # epoch number
        self.num_epoch = num_epoch

        # define check point info
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_step = checkpoint_step

        # define logger
        self.logger_proj_name = logger_proj_name
        self.test_logger_path = test_logger_path

        # init parallel
        self.parallel = parallel
        if self.parallel is not None:
            if self.parallel=="DDP":
                print("[-----pretrain.py]Enable DDP mode")
                self.init_DDP()
            print(f"[-----pretrain.py]Distributed device: {torch.cuda.current_device()} (global rank {dist.get_rank()}/{dist.get_world_size()})")

            # init monitor on device 0
            if dist.get_rank() == 0 and self.logger_proj_name is not None:
                print(f"[-----pretrain.py]wandb logg on device 0")
                wandb.init(project=self.logger_proj_name)
        else:
            print("[-----pretrain.py]Disable parallel mode")
            self.model = self.model.to(self.device)
            self.model.device = self.device

            # init monitor
            if self.logger_proj_name is not None:
                wandb.init(project=self.logger_proj_name)

        #init dataloader
        self.DataLoader_config['dataset'] = self.Dataset
        self.DataLoader_config['drawable_indices'] = self.drawableidx_train
        self.DataLoader_config['distributed'] = False if self.parallel is None else True
        self.DataLoader_use = DataLoader_use
        if self.DataLoader_use == 'MultiomicLoader':
            self.DataLoader = MultiomicLoader(**self.DataLoader_config)
        elif self.DataLoader_use == 'MultiomicLoader_stat':
            self.DataLoader = MultiomicLoader_stat(**self.DataLoader_config)

        #init grad clip
        self.grad_clip = grad_clip
        if self.grad_clip:
            print("[-----pretrain.py]enable grad clip with zclip")
            self.zclip = ZClip(mode="zscore", alpha=0.97, z_thresh=2.5, clip_option="adaptive_scaling", max_grad_norm=1.0,
                          clip_factor=1.0)

    def init_DDP(self):
        # init nccl
        dist.init_process_group(backend="nccl")

        # get local rank
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        print(f"[-----pretrain.py]Current local rank:{self.local_rank}")

        # set device according to rank
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        print(f"[-----pretrain.py]Reset device to local rank: {self.device}")

        # move the model to device
        self.model.to(self.device)
        self.model.device = self.device

        # get DDP
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=True
        )

        #add some info into model
        self.model.task = self.model.module.task
        self.model.pad_id = self.model.module.pad_id

    def test_dataset(self):
        start = time.time()
        print(self.Dataset[10000])
        print("[-----pretrain.py]Time to a single sample:", time.time() - start)

    def test_dataloader(self):
        start = time.time()
        for RNA_module, ATAC_module in self.DataLoader:
            print("[-----pretrain.py]Fetched a batch")
            break
        print("[-----pretrain.py]Time to fetch first batch:", time.time() - start)

    def test_minibatch(self,step=50,ifreturn=True):
        count = 1
        RNA_data = {}
        ATAC_data = {}
        for RNA_module, ATAC_module in self.DataLoader:
            print(f"--------------step {count}------------------")
            RNA_data[count-1] = RNA_module
            for key, tensor in RNA_module.items():
                print(f"{key}: {tensor.dtype}")
            ATAC_data[count-1] = ATAC_module
            for key, tensor in ATAC_module.items():
                print(f"{key}: {tensor.dtype}")

            count = count+1
            if count == step:
                if ifreturn:
                    return RNA_data, ATAC_data
                break

    def test_drawableidx_distributed(self):
        print(f'Current device rank: {self.local_rank}')
        print(f'Number of minibatches on this device: {self.DataLoader.num_batches}')
        print(f'First 30 of all drawable idx: {self.DataLoader.sampler.drawable_indices[0:30]}')

        temp1 = []
        count = 0
        for i in self.DataLoader.sampler:
            temp1.append(i)
            count = count+1
            if count == 10:
                break
        print(f'First 10 of drawable idx on this device: {temp1}')


    def test_forward(self,ifreturn=True):
        torch.cuda.empty_cache()
        for RNA_module, ATAC_module in self.DataLoader:
            for key, tensor in RNA_module.items():
                RNA_module[key] = tensor.to(self.device)
            for key, tensor in ATAC_module.items():
                ATAC_module[key] = tensor.to(self.device)

            # Assume `model` is your PyTorch model and `input_data` is your input tensor.
            reporter = MemReporter(self.model)
            out_RNA, out_ATAC = self.model(RNA_module, ATAC_module)
            reporter.report()

            print(f'[-----pretrain.py]output RNA of the minibatch:')
            print(f'{out_RNA}\nshape: {out_RNA.shape}')
            print(f'[-----pretrain.py]output ATAC of the minibatch:')
            print(f'{out_ATAC}\nshape: {out_ATAC.shape}')

            if ifreturn:
                return out_RNA, out_ATAC

            break

    def test_train(self, step=2, ifreturn=True):
        self.model.train()
        count = 1
        # start train
        for RNA_module, ATAC_module in self.DataLoader:
            self.train_single_MSELoss(RNA_module, ATAC_module, 1, count, None)
            count = count + 1
            if count > step:
                break

    def setup_logger(self,rank):
        # Create a logger for the current process
        logger = logging.getLogger(f'gpu_{rank}')
        logger.setLevel(logging.DEBUG)

        # Create a file handler that logs debug and higher level messages
        file_handler = logging.FileHandler(os.path.join(self.test_logger_path,f'gpu_{rank}.log'))
        file_handler.setLevel(logging.DEBUG)

        # Optional: Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the file handler to the logger
        logger.addHandler(file_handler)

        return logger

    def test_seed(self):
        self.model.train()
        rank = dist.get_rank()  # Each process gets its rank
        logger = self.setup_logger(rank)

        # print seed on each gpu
        logger.info(f"[-----pretrain.py GPU {dist.get_rank()}]CPU seed: {torch.initial_seed()}")
        logger.info(f"[-----pretrain.py GPU {dist.get_rank()}]GPU seed: {torch.cuda.initial_seed()}")

        # check whether equal of params across gpus
        ifequal = self.check_weights_equal()
        logger.info(
            f"[-----pretrain.py GPU {dist.get_rank()}]Params are{' NOT' if not ifequal else ''} equal across GPUs")

        # print first few initialized parameter on each GPU
        for name, param in self.model.named_parameters():
            # Only print parameters that are trainable
            if param.requires_grad:
                # Print the first few values from this parameter tensor
                logger.info(f"[-----pretrain.py GPU {dist.get_rank()}]Initialized model para of {name}: {param.data.view(-1)[:5]}")
                # Optionally, break after printing the first parameter group
                break

    def check_weights_equal(self,baseline = None,atol=1e-6):
        world_size = dist.get_world_size()

        #para list [[para1 of gpu1,para1 of gpu2,...],
        #                [para2 of gpu1,para2 of gpu2,...],...]
        para_list = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Detach and get a CPU copy (if desired) of the parameter tensor.
                param_tensor = param.data
                # Create a list to gather the parameter tensor from all GPUs.
                gathered_tensors = [torch.zeros_like(param_tensor) for _ in range(world_size)]

                # Gather the parameter from all processes.
                dist.all_gather(gathered_tensors, param_tensor)

                # get list
                para_list.append(gathered_tensors)

        # Use the first GPU's tensor as the baseline, or mannually set the baseline.
        # [para1,para2,...]
        if baseline is None:
            baseline = [x[0] for x in para_list]

        for i in range(len(baseline)):
            baseline_param = baseline[i] # baseline param i
            param_allgpus = para_list[i] # baseline param i on all gpus
            for idx, param_eachgpu in enumerate(param_allgpus[1:], start=1):
                if not torch.allclose(baseline_param, param_eachgpu, atol=atol):
                    return False
        return True

    def test_DDP(self, max_step = 10):
        self.model.train()
        rank = dist.get_rank()  # Each process gets its rank
        logger = self.setup_logger(rank)
        count = 1

        # record the original weights before training
        param_tensor_first = [param.data.clone() for _,param in self.model.named_parameters() if param.requires_grad]

        # start test
        for RNA_module, ATAC_module in self.DataLoader:
            logger.info(f"[-----pretrain.py GPU {dist.get_rank()}]-------------STEP{count}-------------------------------")
            # record the weights before update
            if count == 1:
                param_tensor_laststep = param_tensor_first
            else:
                param_tensor_laststep = [param.data.clone() for _, param in self.model.named_parameters() if param.requires_grad]

            # Move data to the GPU
            for key, tensor in RNA_module.items():
                RNA_module[key] = tensor.to(self.device)
            for key, tensor in ATAC_module.items():
                ATAC_module[key] = tensor.to(self.device)

            with autocast(enabled=True):
                # Forward pass
                out_RNA, out_ATAC = self.model(RNA_module, ATAC_module)
                # Compute the loss
                temp_res = torch.concat([out_RNA, out_ATAC], dim=1)
                temp_mask = torch.concat([RNA_module['gExpr_mask'], ATAC_module['aExpr_mask']], dim=1).bool()
                predict = temp_res[temp_mask]
                target = torch.concat([RNA_module['gExpr'], ATAC_module['aExpr']], dim=1).unsqueeze(-1)[temp_mask]
                loss = self.loss_func(predict, target)

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.optimizer.param_groups[0]['lr'] > 1e-8:
                self.scheduler.step()
            logger.info(f"[-----pretrain.py GPU {dist.get_rank()}]Step {count} Loss: {loss.item():.4f}")

            # check update of params compared to last step
            ifnochange = self.check_weights_equal(baseline=param_tensor_laststep)
            logger.info(f"[-----pretrain.py GPU {dist.get_rank()}]Params are{' NOT' if ifnochange else ''} updated compared to last step")

            # check whether equal of params across gpus
            ifequal = self.check_weights_equal()
            logger.info(f"[-----pretrain.py GPU {dist.get_rank()}]Params are{' NOT' if not ifequal else ''} equal across GPUs")

            count = count + 1
            if count > max_step:
                break

    def validation(self, model_w_path, valid_epoch_num = 1):
        print(f"[-----pretrain.py]Enabling validation mode")
        print(f'[-----pretrain.py]validation data size (drawableidx_train): {len(self.drawableidx_valid)})')

        # init train dataloader
        self.DataLoader_config_valid = self.DataLoader_config.copy()
        self.DataLoader_config_valid['drawable_indices'] = self.drawableidx_valid
        if self.DataLoader_use == 'MultiomicLoader':
            self.DataLoader_valid = MultiomicLoader(**self.DataLoader_config_valid)
        elif self.DataLoader_use == 'MultiomicLoader_stat':
            self.DataLoader_valid = MultiomicLoader_stat(**self.DataLoader_config_valid)

        # load model w to be tested
        print(f"[-----pretrain.py]load previous trained weights from {model_w_path}")
        checkpoint = torch.load(model_w_path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        self.model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            total_loss = []
            for epoch in range(valid_epoch_num):
                count = 1
                print(f'[-----pretrain.py]Validation {epoch + 1}/{valid_epoch_num}')

                # set the epoach for the dataloader. This ensures that the shuffling is different across epochs
                self.DataLoader_valid.set_epoch(epoch)

                # # start train
                for RNA_module, ATAC_module in self.DataLoader_valid:
                    # Move data to the GPU
                    # print('move to GPU')
                    for key, tensor in RNA_module.items():
                        RNA_module[key] = tensor.to(self.device)
                    for key, tensor in ATAC_module.items():
                        ATAC_module[key] = tensor.to(self.device)

                    with autocast(enabled=True):
                        # Forward pass
                        # print('forward')
                        out_RNA, out_ATAC = self.model(RNA_module, ATAC_module)

                        # Compute the infoNCE loss
                        # print('loss')
                        if self.model.task == 'random_NTP':
                            RNApad_mask = (RNA_module['gID'] != self.model.pad_id).unsqueeze(-1)
                            ATACpad_mask = (ATAC_module['aPos'] != 0).unsqueeze(-1)
                            temp_mask = torch.concat([RNApad_mask, ATACpad_mask], dim=1)
                            predict = torch.concat([out_RNA, out_ATAC], dim=1)[temp_mask]
                            target = torch.concat([RNA_module['gExpr'], ATAC_module['aExpr']], dim=1).unsqueeze(-1)[
                                temp_mask]
                            loss = self.loss_func(predict, target)
                        else:
                            temp_res = torch.concat([out_RNA, out_ATAC], dim=1)
                            temp_mask = torch.concat([RNA_module['gExpr_mask'], ATAC_module['aExpr_mask']],
                                                     dim=1).bool()
                            predict = temp_res[temp_mask]
                            target = torch.concat([RNA_module['gExpr'], ATAC_module['aExpr']], dim=1).unsqueeze(-1)[
                                temp_mask]
                            loss = self.loss_func(predict, target)
                    print(f"[-----pretrain.py]Step {count} Loss: {loss.item():.8f}")
                    total_loss.append(loss.item())  # Sum the loss
                    count = count + 1

        print(f"[-----pretrain.py]Validation complete, average loss: {np.mean(total_loss)}")
        return total_loss


    def train_single_MSELoss(self, RNA_module, ATAC_module, epoch, count, epoch_checkpoint_dir):
        torch.cuda.empty_cache()
        # Move data to the GPU
        # print('move to GPU')
        for key, tensor in RNA_module.items():
            RNA_module[key] = tensor.to(self.device)
        for key, tensor in ATAC_module.items():
            ATAC_module[key] = tensor.to(self.device)

        with autocast(enabled=True):
            # Forward pass
            # print('forward')
            out_RNA, out_ATAC, order = self.model(RNA_module, ATAC_module)

            # Compute the infoNCE loss
            # print('loss')
            if self.model.task == 'random_NTP':
                RNApad_mask = (RNA_module['gID'] != self.model.pad_id).unsqueeze(-1)
                ATACpad_mask = (ATAC_module['aPos'] != 0).unsqueeze(-1)
                if order == 0:
                    RNApad_mask[:, :1024, :] = False  # do not calculate the loss with first 1024 tokens
                else:
                    ATACpad_mask[:, :1024, :] = False  # do not predict the first 1024 tokens
                temp_mask = torch.concat([RNApad_mask, ATACpad_mask], dim=1)
                predict = torch.concat([out_RNA, out_ATAC], dim=1)[temp_mask]
                target = torch.concat([RNA_module['gExpr'], ATAC_module['aExpr']], dim=1).unsqueeze(-1)[temp_mask]
                loss = self.loss_func(predict, target)

            else:
                temp_res = torch.concat([out_RNA, out_ATAC], dim=1)
                temp_mask = torch.concat([RNA_module['gExpr_mask'], ATAC_module['aExpr_mask']], dim=1).bool()
                predict = temp_res[temp_mask]
                target = torch.concat([RNA_module['gExpr'], ATAC_module['aExpr']], dim=1).unsqueeze(-1)[temp_mask]
                loss = self.loss_func(predict, target)

        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        if self.grad_clip:
            self.scaler.unscale_(self.optimizer)
            self.zclip.step(self.model)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.optimizer.param_groups[0]['lr'] > 1e-8:
            self.scheduler.step()
        print(f"[-----pretrain.py]Step {count} Loss: {loss.item():.4f}")

        # Saving check point
        if self.checkpoint_dir != None and (count % self.checkpoint_step == 0 or count == self.DataLoader.num_batches):
            if self.parallel == "DDP":
                # save with torch
                print(f'[-----pretrain.py]Saving checkpoint at step {count}')
                checkpoint = {
                    'epoch_idx': epoch,
                    'step_idx': count - 1,
                    'model_state_dict': self.model.module.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                file_path = os.path.join(epoch_checkpoint_dir, f'Step_idx_{count - 1}.pth')
                torch.save(checkpoint, file_path)
            else:
                # save with torch
                print(f'[-----pretrain.py]Saving checkpoint at step {count}')
                checkpoint = {
                    'epoch_idx': epoch,
                    'step_idx': count - 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'loss': loss.item(),
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                file_path = os.path.join(epoch_checkpoint_dir, f'Step_idx_{count - 1}.pth')
                torch.save(checkpoint, file_path)

        # log with wandb
        if self.parallel != None:
            if self.logger_proj_name != None and dist.get_rank() == 0:
                # document several metric:
                res_stat = self.Dataset.input_stat(RNA_module, ATAC_module)
                res_stat['loss'] = loss.item()
                res_stat['lr'] = self.optimizer.param_groups[0]['lr']
                wandb.log(res_stat)
        else:
            if self.logger_proj_name != None:
                # document several metric:
                res_stat = self.Dataset.input_stat(RNA_module, ATAC_module)
                res_stat['loss'] = loss.item()
                res_stat['lr'] = self.optimizer.param_groups[0]['lr']
                wandb.log(res_stat)


    def train(self):
        """training"""
        self.model.train()
        for epoch in range(self.num_epoch):
            count = 1
            print(f'[-----pretrain.py]Epoch {epoch+1}/{self.num_epoch}')

            # create a dir for check point of this epoch
            epoch_checkpoint_dir = os.path.join(self.checkpoint_dir,f'epoch_{epoch+1}')
            if not os.path.exists(epoch_checkpoint_dir) and dist.get_rank() == 0:
                os.makedirs(epoch_checkpoint_dir)

            #set the epoach for the dataloader. This ensures that the shuffling is different across epochs
            self.DataLoader.set_epoch(epoch)

            # # start train
            for RNA_module, ATAC_module in self.DataLoader:
                self.train_single_MSELoss(RNA_module, ATAC_module, epoch, count, epoch_checkpoint_dir)
                count = count + 1


        print("[-----pretrain.py]Training complete!")
        if self.logger_proj_name != None:
            wandb.finish()




