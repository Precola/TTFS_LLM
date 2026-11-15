import torch.nn as nn
import torch
from torch.nn.init import normal_
import torch.nn.functional as F
from .base_TTFS import TTFS_block_sum
from .basic_layer_TTFS import LayerNorm, TTFS_dropout,  TTFS_unit

class OneHotEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, initializer_range=0.02, use_bias=False, X_n=0.64):
        super().__init__()
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.B_n = X_n

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(num_embeddings, embedding_dim, bias=use_bias)#
        normal_(self.linear.weight, mean=0, std=initializer_range)
        if use_bias:
            nn.init.zeros_(self.linear.bias)

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer. Alpha is fixed at 1.
        """
        self.t_min_prev = t_min_prev
        self.t_min = t_min
        self.t_max =t_min + self.B_n
        return self.t_min, self.t_max

    def forward(self, input_ids):
        """
        input_ids: LongTensor of shape (batch_size,) or (batch_size, seq_len)
        returns: (batch_size, embedding_dim) or (batch_size, seq_len, embedding_dim)
        """
        one_hot = F.one_hot(input_ids, num_classes=self.num_embeddings).float() # spike time
        z = self.linear(one_hot - self.t_min) + self.t_max
        t_i = torch.where(z < self.t_max + self.B_n, z, self.t_max + self.B_n)

        return t_i

class PositionalEmbedding(nn.Module):
    """
    Note: The positional encoding used in BERT is fundamentally different from that in the original Transformer.
In BERT, positional encoding is implemented as a learnable embedding layer, while in the Transformer paper, it's computed using sinusoidal functions based on fixed formulas.

      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
                                                 ————————  GoogleResearch
    https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/modeling.py
    """

    def __init__(self, hidden_size, max_position_embeddings=512, initializer_range=0.02):
        super(PositionalEmbedding, self).__init__()
        self.embedding = OneHotEmbedding(max_position_embeddings, hidden_size, initializer_range=0.02)
        self._reset_parameters(initializer_range)

    def forward(self, position_ids):
        """
        :param position_ids: [1,position_ids_len]
        :return: [position_ids_len, 1, hidden_size]
        """
        return self.embedding(position_ids).transpose(0, 1)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""

        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, initializer_range=0.02):
        super(TokenEmbedding, self).__init__()
        self.embedding = OneHotEmbedding(vocab_size, hidden_size, initializer_range=0.02)
        self._reset_parameters(initializer_range)

    def forward(self, input_ids):
        """
        :param input_ids: shape : [input_ids_len, batch_size]
        :return: shape: [input_ids_len, batch_size, hidden_size]
        """
        return self.embedding(input_ids)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class SegmentEmbedding(nn.Module):
    def __init__(self, type_vocab_size, hidden_size, initializer_range=0.02):
        super(SegmentEmbedding, self).__init__()
        self.embedding = OneHotEmbedding(type_vocab_size, hidden_size, initializer_range=0.02)
        self._reset_parameters(initializer_range)

    def forward(self, token_type_ids):
        """

        :param token_type_ids:  shape: [token_type_ids_len, batch_size]
        :return: shape: [token_type_ids_len, batch_size, hidden_size]
        """
        return self.embedding(token_type_ids)

    def _reset_parameters(self, initializer_range):
        r"""Initiate parameters."""
        """
        初始化
        """
        for p in self.parameters():
            if p.dim() > 1:
                normal_(p, mean=0.0, std=initializer_range)


class BertEmbeddings(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : normal embedding matrix
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = TokenEmbedding(vocab_size=config.vocab_size,
                                              hidden_size=config.hidden_size,
                                              initializer_range=config.initializer_range)
        # return shape [src_len,batch_size,hidden_size]

        self.position_embeddings = PositionalEmbedding(max_position_embeddings=config.max_position_embeddings,
                                                       hidden_size=config.hidden_size,
                                                       initializer_range=config.initializer_range)
        # return shape [src_len,1,hidden_size]

        self.token_type_embeddings = SegmentEmbedding(type_vocab_size=config.type_vocab_size,
                                                      hidden_size=config.hidden_size,
                                                      initializer_range=config.initializer_range)
        # return shape  [src_len,batch_size,hidden_size]

        self.sum_layer = TTFS_block_sum(3, 1)
        self.LayerNorm = LayerNorm(config.hidden_size, bias=False, eps=1e-12) # TODO
        self.dropout = TTFS_dropout(config.hidden_dropout_prob)

        self.layers = [self.sum_layer, self.LayerNorm, self.dropout]

        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings).expand((1, -1)))
        # shape: [1, max_position_embeddings]

    def set_params(self, t_min_prev, t_min):
        self.word_embeddings.embedding.set_params(t_min_prev, t_min)
        self.position_embeddings.embedding.set_params(t_min_prev, t_min)
        t_min, t_max = self.token_type_embeddings.embedding.set_params(t_min_prev, t_min)

        for l in self.layers:
            t_tuple = l.set_params(t_min, t_max)
            t_min, t_max = t_tuple[-2:]

        return t_min, t_max

    def forward(self,
                input_ids=None,
                position_ids=None,
                token_type_ids=None):
        """
        :param input_ids:  输入序列的原始token id, shape: [src_len, batch_size]
        :param position_ids: 位置序列，本质就是 [0,1,2,3,...,src_len-1], shape: [1,src_len]
        :param token_type_ids: 句子分隔token, 例如[0,0,0,0,1,1,1,1]用于区分两个句子 shape:[src_len,batch_size]
        :return: [src_len, batch_size, hidden_size]
        """
        src_len = input_ids.size(0)
        token_embedding = self.word_embeddings(input_ids)
        # shape:[src_len,batch_size,hidden_size]

        if position_ids is None:  # 在实际建模时这个参数其实可以不用传值
            position_ids = self.position_ids[:, :src_len]  # [1,src_len]
        positional_embedding = self.position_embeddings(position_ids)
        # [src_len, 1, hidden_size]

        if token_type_ids is None:  # 如果输入模型的只有一个序列，那么这个参数也不用传值
            token_type_ids = torch.zeros_like(input_ids,
                                              device=self.position_ids.device)  # [src_len, batch_size]
        segment_embedding = self.token_type_embeddings(token_type_ids)
        # [src_len,batch_size,hidden_size]

        positional_embedding = positional_embedding.expand(-1, segment_embedding.shape[1], -1)
        embeddings = torch.stack([token_embedding, positional_embedding, segment_embedding], dim=-1)
        embeddings = self.sum_layer(embeddings)
        embeddings = embeddings.squeeze(-1)
        # [src_len,batch_size,hidden_size] + [src_len,1,hidden_size] + [src_len,batch_size,hidden_size]
        embeddings = self.LayerNorm(embeddings)  # [src_len, batch_size, hidden_size]
        embeddings = self.dropout(embeddings)
        return embeddings
