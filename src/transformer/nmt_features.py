# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.
import torch
from torch.autograd import Variable
import numpy as np
from torch.tensor import Tensor
from transformer.transformer_utils import subsequent_mask


class Batch(object):

    def __init__(self, src: Tensor, target: Tensor, pad=0):
        """
        @param src shape = batch_size, max_length
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2) # batch_size, 1, max_length
        if target is not None:
            # assume target is " <start> A B C D"
            self.target = target[:, :-1] # <start> A B C
            self.target_y = target[:, 1:] # A B C D
            self.target_mask = self.make_std_mask(self.target, pad)

            self.number_tokens = (self.target_y != pad).data.sum()

    @staticmethod
    def make_std_mask(target: Tensor, pad):
        target_mask = (target!=pad).unsqueeze(-2)
        target_mask = target_mask & Variable(subsequent_mask(target.size(-1)).type_as(target_mask.data))
        return target_mask


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)