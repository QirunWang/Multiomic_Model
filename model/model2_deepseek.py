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


# input
# [g1, ..., gn] [btz,token_len]
# [x1, ..., xn] [btz,token_len]
# [p1, ..., pn] [btz,token_len]
# [s1, ..., sn] [btz,token_len]
# dist matrix [btz, [token_len,token_len]]








