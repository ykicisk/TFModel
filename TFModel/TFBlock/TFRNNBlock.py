#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from TFBlockBase import TFBlockBase
from TFLinearBlock import TFLinearBlock


class TFRNNBlock(TFBlockBase):
    u"""二次元tf.Variableに対して線形レイヤ連打"""
    def __init__(self, src_op, forget_bias, n_unit, embedding_dims=None,
                 name="RNNBlock"):
        u"""ここの__init__で self.dst_op(出力Operator)を定義する
        """
        super(TFRNNBlock, self).__init__(name)
        with tf.variable_scope(name):
            len_seq = src_op.get_shape().dims[1].value
            n_src_dim = src_op.get_shape().dims[2].value
            # (batch, len_seq, src_dim) => (len_seq, batch, src_dim)
            in1 = tf.transpose(src_op, [1, 0, 2])
            # (len_seq, batch, src_dim) => (len_seq * batch, src_dim)
            in2 = tf.reshape(in1, [-1, n_src_dim])
            # embedding
            if embedding_dims is None:
                embedding_dims = [n_unit]
            else:
                embedding_dims.append(n_unit)
            self.ebd_block = TFLinearBlock(src_op=in2,
                                           n_dims=embedding_dims,
                                           dropout_rate=0.2)
            in3 = self.ebd_block.get_dst_op()
            # (len_seq * batch, src_dim) => (len_seq, batch, src_dim)
            rnn_src_op = tf.split(0, len_seq, in3)
            # RNN
            # daily, weekly, monthlyのRNNを別扱いにする。
            # 一部のNN構成要素はscopeを明示的に変更する必要がある
            self.cell = tf.nn.rnn_cell.BasicLSTMCell(n_unit,
                                                     forget_bias=forget_bias)
            self.init_state_ph = tf.placeholder(tf.float32,
                                                [None, self.cell.state_size],
                                                name="init_state_ph")
            with tf.variable_scope(self.name):
                rnn_dst_op, _ = tf.nn.rnn(self.cell, rnn_src_op,
                                          initial_state=self.init_state_ph,
                                          dtype=tf.float32)
            self.dst_op = rnn_dst_op[-1]

    def get_const_placeholders(self, batch_size, train_flag=False):
        u"""学習データとは別にconst値のplaceholderがあれば返す.
        例) RNNの初期state, keep_prob等"""
        val = np.zeros((batch_size, self.cell.state_size))
        ret = {self.init_state_ph.name: val}
        ret.update(self.ebd_block.get_const_placeholders(batch_size,
                                                         train_flag))
        return ret
