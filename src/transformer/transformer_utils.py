# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.

import torch
import math
from torch.nn import Module, ModuleList
from torch.tensor import Tensor
import torch.nn.functional as F
import torch.nn as nn
import copy
import numpy as np

def clones(module: Module, n: int) -> ModuleList:
    """
    Produce n identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def attention(query: Tensor, key: Tensor, value:Tensor, mask:Tensor=None, dropout=None):
    """
    scaled dot production attention
    @param query shape -> batch_size, head_count, max_length, model_dim_size/head_count
    @param key shape -> batch_size, head_count, max_length, model_dim_size/head_count
    @param value shape -> batch_size, head_count, max_length, model_dim_size/head_count

    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))/ math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill(mask == 0, -1e9)
    p_attention = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attention = dropout(p_attention)
    return torch.matmul(p_attention, value), p_attention


def subsequent_mask(size) -> Tensor:
    """ return mask tensor in boolean type"""
    attn_shape = (1, size, size)
    subsequent_mask_data = np.triu(np.ones(attn_shape), k=1,).astype('uint8')
    return torch.from_numpy(subsequent_mask_data) == 0





