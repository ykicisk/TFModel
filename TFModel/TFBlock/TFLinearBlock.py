#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from TFBlockBase import TFBlockBase


class TFLinearBlock(TFBlockBase):
    u"""二次元tf.Variableに対して線形レイヤ連打"""
    def __init__(self, src_op, n_dims, dropout_rate=0.2, last_drop=False,
                 name="LinearBlock"):
        u"""ここの__init__で self.dst_op(出力Operator)を定義する
        """
        super(TFLinearBlock, self).__init__(name)
        with tf.variable_scope(name):
            self.dropout_rate = dropout_rate
            self.W_list = []
            self.b_list = []
            self.h_list = []
            self.keep_prob = tf.placeholder("float", name="keep_prob")
            now_op = src_op
            # now_op = tf.Print(now_op, [now_op],"now_op: {name}:src".format(name=self.name), summarize=10)
            for idx, next_dim in enumerate(n_dims):
                before_dim = now_op.get_shape().dims[1].value
                # print idx, before_dim, next_dim
                W_name = "W_{i}".format(name=name, i=idx)
                W_init = tf.truncated_normal([before_dim, next_dim], stddev=1.0)
                W = tf.Variable(W_init, name=W_name)
                # W = tf.Print(W, [W],"W({name}:{idx}) :".format(name=self.name, idx=idx), summarize=10)
                b_name = "b_{i}".format(name=name, i=idx)
                b_init = tf.truncated_normal([next_dim], stddev=1.0)
                b = tf.Variable(b_init, name=b_name)
                self.W_list.append(W)
                self.b_list.append(b)
                h = tf.nn.relu(tf.matmul(now_op, W) + b)
                if last_drop or idx < len(n_dims) - 1:
                    now_op = tf.nn.dropout(h, self.keep_prob)
                else:
                    now_op = h
            self.dst_op = now_op

    def get_const_placeholders(self, batch_size, train_flag=False):
        u"""学習データとは別にconst値のplaceholderがあれば返す.
        例) RNNの初期state, keep_prob等"""
        val = 1.0 - self.dropout_rate if train_flag else 1.0
        return {self.keep_prob.name: val}
