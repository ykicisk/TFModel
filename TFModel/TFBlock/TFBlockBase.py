#!/usr/bin/env python
# -*- coding:utf-8 -*-
from abc import ABCMeta, abstractmethod


class TFBlockBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, name):
        u"""ここの__init__で self.dst_op(出力Operator)を定義する
        """
        self.name = name

    @abstractmethod
    def get_const_placeholders(self, batch_size, train_flag=False):
        u"""学習データとは別にconst値のplaceholderがあれば返す. 
        例) RNNの初期state, keep_prob等"""
        return {}

    def get_dst_op(self):
        return self.dst_op
