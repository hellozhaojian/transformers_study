# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.
from transformer.nmt_features import data_gen, Batch
from transformer.nmt_transformer import run_epoch, MyOptimizer, NmtTransformer, EncoderDecoder, greedy_decode, SimpleLossCompute
from transformer.transformer_layers import LableSmoothingLayer
import torch
from torch.autograd import Variable

def toy_main():
    number_of_class = 11
    model_dim = 512
    # train
    criterion = LableSmoothingLayer(number_of_class,0,0.0)
    model: NmtTransformer = NmtTransformer(number_of_class, number_of_class, model_dim=model_dim)
    model_opt = MyOptimizer(model_dim, 1, 400,   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(10):
        model.train()
        run_epoch(data_gen(number_of_class, 30, 20), model,
                  SimpleLossCompute(model.inner_model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(number_of_class, 30, 5), model,
                        SimpleLossCompute(model.inner_model.generator, criterion, None)))
    # predict
    model.eval()
    src = Variable(torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]))
    src_mask = Variable(torch.ones(1, 1, 10))
    print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))
    pass


if __name__ == "__main__":
    toy_main()