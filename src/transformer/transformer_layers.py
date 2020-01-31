# encoding: utf-8
# Copyright 2019 The DeepNlp Authors.

import numpy as np
import torch
from torch.tensor import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from transformer.transformer_utils import clones, attention
seaborn.set_context(context="talk")


class LinearSoftmaxGenerator(nn.Module):
    """Define standard linear + softmax generation step"""
    def __init__(self, input_dim, output_dim):

        super(LinearSoftmaxGenerator, self).__init__()
        self.project_function = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.log_softmax(self.project_function(x), dim=-1)


class EmbeddingLayer(nn.Module):

    def __init__(self, vocab_size, emb_size):
        super(EmbeddingLayer, self).__init__()
        self.emb_layer = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, x):
        return self.emb_layer(x) * math.sqrt(self.emb_size)


class PositionLayer(nn.Module):

    MODE_EXPAND = "MODE_EXPAND"
    MODE_ADD = "MODE_ADD"
    MODE_CONCAT = "MODE_CONCAT"

    def __init__(self, max_pos, emb_size, mode=MODE_ADD):
        super(PositionLayer, self).__init__()
        self.max_pos = max_pos
        self.emb_size = emb_size
        self.mode = mode
        if mode == PositionLayer.MODE_EXPAND:
            self.weights = nn.Parameter(torch.Tensor(self.max_pos * 2 + 1, self.emb_size))
        else:
            self.weights = nn.Parameter(torch.Tensor(self.max_pos, self.emb_size))
        self.reset_param()

    def reset_param(self):
        torch.nn.init.xavier_normal(self.weights)

    def forward(self, x: torch.Tensor):

        if self.mode == self.MODE_EXPAND:
            indices = torch.clamp(x, -self.max_pos, self.max_pos) + self.max_pos
            return F.embedding(indices.type(torch.LongTensor), self.weights)

        batch_size, seq_len = x.size()[:2]
        assert seq_len <= self.max_pos
        embeddings = self.weights[:seq_len, :].view(1, seq_len, self.emb_size)
        #  x已经是word embedding的时候，这个操作是有效的
        if self.mode == self.MODE_ADD:
            return x + embeddings
        if self.mode == self.MODE_CONCAT:
            return torch.cat((x, embeddings.repeat(batch_size, 1, 1)), dim=-1)
        raise NotImplementedError(f'unknown mode {self.mode}')

    def extra_repr(self) -> str:
        return f"max_pos={self.max_pos}, embedding_dim={self.emb_size} mode={self.mode}"


class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, head_count, model_dim, dropout=0.1):
        super(MultiHeadAttentionLayer, self).__init__()
        self.head_count = head_count
        self.model_dim = model_dim
        assert self.model_dim % self.head_count == 0
        self.model_k_dim = self.model_dim // self.head_count
        self.dropout = torch.nn.Dropout(dropout)
        self.linears: torch.nn.ModuleList = clones(nn.Linear(self.model_dim, self.model_dim), 4)
        self.attn = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        """
        @param query shape -> [batch_size, max_length, emb_size]
        @param key shape -> [batch_size, max_length, emb_size]
        @param value shape -> [batch_size, max_length, emb_size]
        @param mask shape -> [1, max_length, max_length]
        @return a tensor with shape -> ??
        """
        if mask is not None:
            # 1, n, n -> 1, 1, n, n; n is max length of sentence
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # do projection
        query, key, value = [linear_f(x).view(batch_size, -1, self.head_count, self.model_k_dim).transpose(1, 2)
                             for linear_f, x in zip(self.linears, (query, key, value))]
        # do attention
        x, self.attn = attention(query, key, value, mask, self.dropout)
        # do concatenation
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_count * self.model_k_dim)
        return self.linears[-1](x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean)/(std + self.eps) + self.b_2


class LayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(LayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):

    def __init__(self, model_dim, ff_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear_first = nn.Linear(model_dim, ff_dim)
        self.linear_second = nn.Linear(ff_dim, model_dim)

    def forward(self, x: Tensor) -> Tensor:
        self.linear_second(self.dropout(self.linear_first(x)))


class EncoderLayer(nn.Module):

    def __init__(self, model_dim, attention_module: MultiHeadAttentionLayer,
                 feed_forward_module: PositionwiseFeedForward, dropout: float=0.1):
        super(EncoderLayer, self).__init__()
        self.attention_module = attention_module
        self.feed_forward_module = feed_forward_module
        self.connection_layers = clones(LayerConnection(model_dim, dropout), 2)
        self.model_dim = model_dim

    def forward(self, x, mask):
        x = self.connection_layers[0](x, lambda x: self.attention_module(x, x, x, mask))
        return self.connection_layers[1](x, self.feed_forward_module)


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, self_attention_module: MultiHeadAttentionLayer,
                 src_attention_module: MultiHeadAttentionLayer, feed_forward_module: PositionwiseFeedForward,
                 dropout: float=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention_module = self_attention_module
        self.src_attention_module = src_attention_module
        self.feed_forward_module = feed_forward_module
        self.connection_layers = clones(LayerConnection(model_dim, dropout), 3)
        self.model_dim = model_dim

    def forward(self, x: Tensor, memory: Tensor, src_mask: Tensor, target_mask: Tensor):
        x = self.connection_layers[0](x, lambda x: self.self_attention_module(x, x, x, target_mask))
        x = self.connection_layers[1](x, lambda x: self.src_attention_module(x, memory, memory, src_mask))
        return self.connection_layers[2](x, self.feed_forward_module)


class Encoder(nn.Module):
    """ stack EncoderLayer"""
    def __init__(self, encoder_layer: EncoderLayer, number_of_layers: int):
        super(Encoder, self).__init__()
        self.encoder_layers = clones(encoder_layer, number_of_layers)
        self.norm = LayerNorm(encoder_layer.model_dim)

    def forward(self, x, mask):
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """ stack DecoderLayer"""
    def __init__(self, decoder_layer: DecoderLayer, number_of_layers: int):
        super(Decoder, self).__init__()
        self.decode_layers = clones(decoder_layer, number_of_layers)
        self.norm = LayerNorm(decoder_layer.model_dim)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.decode_layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_emb: nn.Module,
                 target_emb: nn.Module, generator: LinearSoftmaxGenerator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.target_emb = target_emb
        self.generator = generator

    def forward(self, src, target, src_mask, target_mask):
        return self.decode(self.encode(src, src_mask), src_mask, target, target_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_emb(src), src_mask)

    def decode(self, memroy, src_mask, target, target_mask):
        return self.decoder(self.target_emb(target), memroy, src_mask, target_mask)
