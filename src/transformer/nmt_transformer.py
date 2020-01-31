# encoding: utf-8
# Copyright 2020 The DeepNlp Authors.
import torch
import torch.nn as nn
import copy
from transformer.transformer_layers import (MultiHeadAttentionLayer, PositionwiseFeedForward,
                                            EmbeddingLayer,PositionLayer, LinearSoftmaxGenerator,
                                            Encoder, Decoder, EncoderDecoder, EncoderLayer, DecoderLayer)


class NmtTransformer(nn.Module):

    def __init__(self, src_vocab_size, target_vocab_size, number_of_layers=6, max_src_pos=512, max_target_pos=512,
                 model_dim=512, forward_dim=2048, head_number=8, dropout=0.1):
        super(NmtTransformer, self).__init__()
        c = copy.deepcopy
        multi_head_attention = MultiHeadAttentionLayer(head_number, model_dim, dropout)
        feed_forward = PositionwiseFeedForward(model_dim, forward_dim, dropout)
        position_src_emb = PositionLayer(max_src_pos, model_dim, dropout)
        position_target_emb = PositionLayer(max_target_pos, model_dim, dropout)
        self.model: nn.Module = EncoderDecoder(
            Encoder(EncoderLayer(model_dim, c(multi_head_attention), c(feed_forward), dropout), number_of_layers),
            Decoder(DecoderLayer(model_dim, c(multi_head_attention), c(multi_head_attention), feed_forward,dropout), number_of_layers),
            nn.Sequential(EmbeddingLayer(src_vocab_size, model_dim), position_src_emb),
            nn.Sequential(EmbeddingLayer(target_vocab_size, model_dim), position_target_emb),
            LinearSoftmaxGenerator(model_dim, target_vocab_size)

        )
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, target, src_mask, target_mask):
        return self.model(src, target, src_mask, target_mask)


if __name__ == "__main__":
    model: NmtTransformer = NmtTransformer(10, 10, 2)
    print(model.model)