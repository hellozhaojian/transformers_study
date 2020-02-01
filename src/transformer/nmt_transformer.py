# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.tensor import Tensor
from torch.autograd import Variable
import copy
from transformer.transformer_layers import (MultiHeadAttentionLayer, PositionwiseFeedForward,
                                            EmbeddingLayer,PositionLayer, LinearSoftmaxGenerator,
                                            Encoder, Decoder, EncoderDecoder, EncoderLayer, DecoderLayer)
from transformer.transformer_utils import (subsequent_mask)
from torch.optim.optimizer import Optimizer
from transformer.nmt_features import Batch
from torch.optim.adam import Adam

class NmtTransformer(nn.Module):

    def __init__(self, src_vocab_size, target_vocab_size, number_of_layers=6, max_src_pos=512, max_target_pos=512,
                 model_dim=512, forward_dim=2048, head_number=8, dropout=0.1):
        super(NmtTransformer, self).__init__()
        c = copy.deepcopy
        multi_head_attention = MultiHeadAttentionLayer(head_number, model_dim, dropout)
        feed_forward = PositionwiseFeedForward(model_dim, forward_dim, dropout)
        position_src_emb = PositionLayer(max_src_pos, model_dim)
        position_target_emb = PositionLayer(max_target_pos, model_dim)
        self.inner_model: EncoderDecoder = EncoderDecoder(
            Encoder(EncoderLayer(model_dim, c(multi_head_attention), c(feed_forward), dropout), number_of_layers),
            Decoder(DecoderLayer(model_dim, c(multi_head_attention), c(multi_head_attention), feed_forward,dropout), number_of_layers),
            nn.Sequential(EmbeddingLayer(src_vocab_size, model_dim), position_src_emb),
            nn.Sequential(EmbeddingLayer(target_vocab_size, model_dim), position_target_emb),
            LinearSoftmaxGenerator(model_dim, target_vocab_size)

        )
        for p in self.inner_model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, target, src_mask, target_mask):
        return self.inner_model(src, target, src_mask, target_mask)


class MyOptimizer(object):

    def __init__(self, model_dim: int, factor: float, warmup: int, optimizer: Adam):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_dim = model_dim
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_dim ** (-0.5) *
                              min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model: EncoderDecoder):
    tmp: EmbeddingLayer = model.src_emb[0]
    return MyOptimizer(tmp.emb_size, 2, 4000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class SimpleLossCompute(object):

    def __init__(self, generator: LinearSoftmaxGenerator, criterion:torch.nn.KLDivLoss, opt: MyOptimizer):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x: Tensor, y: Tensor, norm):
        x: Tensor = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))/norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        print(loss)
        return loss.data.item() * norm


def run_epoch(data_iter, model: NmtTransformer, loss_compute: SimpleLossCompute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, data in enumerate(data_iter):
        batch: Batch = data
        out = model.forward(batch.src, batch.target, batch.src_mask, batch.target_mask)
        loss = loss_compute(out, batch.target_y, batch.number_tokens)
        total_loss += loss
        total_tokens += batch.number_tokens
        tokens += batch.number_tokens
        if i % 50 == 1:
            elapsed = time.time() - start
            avg_loss = loss/(batch.number_tokens + 1)
            tokens_per_second = tokens / (elapsed+0.1)
            print(f"Epoch step : {i} Loss: {avg_loss}, tokens per sec {tokens_per_second}")
            start = time.time()
            tokens = 0
    return total_loss/(total_tokens + 1)


def greedy_decode(model: NmtTransformer, src: Tensor, src_mask: Tensor, max_len: int, start_symbol):
    inner_model: EncoderDecoder = model.inner_model
    memory = inner_model.encode(src, src_mask)
    y_start = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len -1):
        out = inner_model.decode(memory, src_mask, Variable(y_start),
                                 Variable(subsequent_mask(y_start.size(1)).type_as(src.data)))
        prob = inner_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[0]
        y_start = torch.cat([y_start, torch.ones(1,1).type_as(src.data).fill_(next_word)], dim=1)
    return y_start

if __name__ == "__main__":
    #model: NmtTransformer = NmtTransformer(10, 10, 2)
    #print(model.model)
    import numpy as np
    opts = [MyOptimizer(512, 1, 4000, None),
            MyOptimizer(512, 1, 8000, None),
            MyOptimizer(256, 1, 4000, None)]
    plt.plot(np.arange(1, 20000), [[opt.rate(i) for opt in opts] for i in range(1, 20000)])
    plt.legend(["512:4000", "512:8000", "256:4000"])
    plt.show()