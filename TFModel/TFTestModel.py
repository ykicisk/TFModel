#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from TFBlock import TFRNNBlock, TFLinearBlock
from TFModelBase import TFModelBase


class TFTestModelBase(TFModelBase):
    def __init__(self, feature_info, board_logdir, scope_name):
        u"""ここの__init__で以下のものを定義する
        1. batch_genで送られてくるデータのplaceholder
        (下のget_additional_const_placeholderに含むものも)
        2. self.loss_op: optimizerで最小化するlossのオペレータ
        3. self.acc_op: 精度のオペレータ
        4. self.dst_op: 予測結果のオペレータ
        """
        # 必ずはじめにやる
        super(TFTestModelBase, self).__init__(feature_info, board_logdir)

        # 共通部分
        with tf.variable_scope(scope_name):
            feature_elems = ["daily", "weekly", "monthly"]
            # ラベルは時系列情報を削除
            # [None, len_seq, n_dim] => [None, 1, n_dim] => [None, n_dim]
            # self.label_op = tf.reshape(tf.split(1,
            #                                     feature_info["label"]["shape"][0],
            #                                     self.feed_ph["label"])[0],
            #                            [-1, 1])
            # print "-- resize label op --"
            # print self.feed_ph["label"].get_shape()
            # print "↓"
            # print self.label_op.get_shape()
            # print "---------------------"

            # 中間データ側で考慮したので不要
            self.label_op = self.feed_ph["label"]
            print "label_op shape:", self.label_op.get_shape()

            # inference周り (dst_op)
            # RNN block(per fgrpu) => fc1まで共通
            self.blocks = {}
            for fgroup in feature_elems:
                with tf.variable_scope(fgroup):
                    block = TFRNNBlock(src_op=self.feed_ph[fgroup],
                                       n_unit=12, embedding_dims=[12],
                                       forget_bias=0.3)
                    self.blocks["{0}_rnn".format(fgroup)] = block
            self.fc1_op = tf.concat(1, [b.get_dst_op()
                                        for b in self.blocks.values()])

    def get_const_placeholders(self, batch_size, train_flag=False):
        u"""学習データとは別にconst値のplaceholderがあれば返す.
        例) RNNの初期state, keep_prob等"""
        ret = {}
        for block in self.blocks.values():
            ret.update(block.get_const_placeholders(batch_size, train_flag))
        return ret


class TFTestClassificationModel(TFTestModelBase):
    def __init__(self, feature_info, board_logdir, keep_label_th=0.02,
                 label_weighting=True):
        u"""ここの__init__で以下のものを定義する
        1. batch_genで送られてくるデータのplaceholder
        (下のget_additional_const_placeholderに含むものも)
        2. self.loss_op: optimizerで最小化するlossのオペレータ
        3. self.acc_op: 精度のオペレータ
        4. self.dst_op: 予測結果のオペレータ

        keep_label_th: 現状維持のしきい値。0のとき実質2値分類になる。
        """
        self.up_cnt = 1  # ゼロ割を防ぐため
        self.down_cnt = 1  # ゼロ割を防ぐため
        self.other_cnt = 1  # ゼロ割を防ぐため
        self.keep_label_th = keep_label_th
        self.label_weighting_flag = label_weighting
        # 共通の設定
        name = "TFTestClassificationModel"
        super(TFTestClassificationModel, self).__init__(
            feature_info, board_logdir, name)
        with tf.variable_scope(name):
            dst_dim = 3 if self.keep_label_th > 0.0 else 2
            self.blocks["fc_linear"] = TFLinearBlock(src_op=self.fc1_op,
                                                     name="fc",
                                                     n_dims=[18, dst_dim],
                                                     dropout_rate=0.3)
            # クラス重み
            self.weight_pf = tf.placeholder(tf.float32, [dst_dim],
                                            name="weight_pf")
        # classificationの場合はsoftmaxをかける
        fc2 = self.blocks["fc_linear"].get_dst_op()
        self.dst_op = tf.nn.softmax(fc2)
        # self.dst_op = tf.Print(self.dst_op, [self.dst_op],"dst_op: ",
        #                        summarize=100)

        # loss_op : ラベルを [1, 0], [0, 1]に変換
        if self.keep_label_th > 0.0:
            th_up = tf.ones_like(self.label_op, tf.float32)*self.keep_label_th
            th_down = tf.ones_like(self.label_op, tf.float32)*(-self.keep_label_th)
            cl_up = tf.cast(tf.greater(self.label_op, th_up), tf.float32)
            cl_down = tf.cast(tf.less(self.label_op, th_down), tf.float32)
            cl_other = tf.ones_like(self.label_op, tf.float32)-cl_up-cl_down
            classification_label = tf.concat(1, [cl_up, cl_other, cl_down])
        else:
            th_up = tf.zeros_like(self.label_op, tf.float32)
            cl_up = tf.cast(tf.greater(self.label_op, th_up), tf.float32)
            cl_down = tf.ones_like(self.label_op, tf.float32) - cl_up
            classification_label = tf.concat(1, [cl_up, cl_down])

        # Trainデータの偏りを考慮してクラス重みを決定
        weighted_cl_label = tf.mul(classification_label, self.weight_pf)
        # weighted_cl_label = tf.Print(weighted_cl_label,
        #                                 [weighted_cl_label],
        #                                 "weighted_cl_label: ",
        #                                 summarize=100)
        tmp = -tf.reduce_sum(weighted_cl_label * tf.log(self.dst_op+1e-50),
                             reduction_indices=[1])
        self.loss_op = tf.reduce_mean(tmp)

        # accuracy_op
        argmax_dst_op = tf.argmax(self.dst_op, 1)
        # self.dst_op = tf.Print(self.dst_op, [self.dst_op],
        #                        "dst_op: ", summarize=10)
        # argmax_dst_op = tf.Print(argmax_dst_op, [argmax_dst_op],
        #                          "argmax_dst_op: ", summarize=10)
        correct = tf.equal(argmax_dst_op,
                           tf.argmax(classification_label, 1))
        self.acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))

    def train_pre_processing(self, train_dict):
        n_u = np.count_nonzero((train_dict["label:0"] > self.keep_label_th)*1)
        n_d = np.count_nonzero((train_dict["label:0"] < -self.keep_label_th)*1)
        n_o = train_dict["label:0"].size - n_u - n_d
        self.up_cnt += n_u
        self.down_cnt += n_d
        self.other_cnt += n_o

    def print_test_summary(self, dst, label):
        n_sample = 10
        if self.label_weighting_flag:
            print "--- class weight ---"
            if self.keep_label_th > 0.0:
                print (self.up_cnt, self.other_cnt, self.down_cnt)
                norm_term = 1.0/self.up_cnt + 1.0/self.other_cnt \
                    + 1.0/self.down_cnt
                print np.array([1.0/self.up_cnt, 1.0/self.other_cnt,
                                1.0/self.down_cnt]) / norm_term
            else:
                print (self.up_cnt, self.down_cnt)
                norm_term = 1.0/self.up_cnt + 1.0/self.down_cnt
                print np.array([1.0/self.up_cnt,
                                1.0/self.down_cnt]) / norm_term
        print "--- dst ---"
        print dst[0:n_sample]
        # label: [None, n_seq, 1]
        print "--- label ---"
        print label[0:n_sample, 0]

    def get_const_placeholders(self, batch_size, train_flag=False):
        u"""学習データとは別にconst値のplaceholderがあれば返す.
        例) RNNの初期state, keep_prob等"""
        ret = super(TFTestClassificationModel, self).get_const_placeholders(
                   batch_size, train_flag
                   )
        # ラベルweightを追加
        pf_name = self.weight_pf.name
        if self.keep_label_th > 0.0:
            if train_flag and self.label_weighting_flag:
                norm_term = 1.0/self.up_cnt + 1.0/self.down_cnt \
                    + 1.0/self.other_cnt
                ret[pf_name] = np.array([1.0/self.up_cnt, 1.0/self.other_cnt,
                                         1.0/self.down_cnt]) / norm_term
            else:
                ret[pf_name] = np.array([1.0, 1.0, 1.0])
        else:
            if train_flag and self.label_weighting_flag:
                norm_term = 1.0/self.up_cnt + 1.0/self.down_cnt
                ret[pf_name] = np.array([1.0/self.up_cnt,
                                         1.0/self.down_cnt]) / norm_term
            else:
                ret[pf_name] = np.array([1.0, 1.0])
        return ret


class TFTestRegressionModel(TFTestModelBase):
    def __init__(self, feature_info, board_logdir, correct_dis_th=0.01):
        u"""ここの__init__で以下のものを定義する
        1. batch_genで送られてくるデータのplaceholder
        (下のget_additional_const_placeholderに含むものも)
        2. self.loss_op: optimizerで最小化するlossのオペレータ
        3. self.acc_op: 精度のオペレータ
        4. self.dst_op: 予測結果のオペレータ
        """
        # ラベルは1次元にする。 label.shape = [len_seq, n_dim]
        dst_dim = feature_info["label"]["shape"][1]
        # 共通の設定
        name = "TFTestRegressionModel"
        super(TFTestRegressionModel, self).__init__(
            feature_info, board_logdir, name)
        with tf.variable_scope(name):
            self.blocks["fc_linear"] = TFLinearBlock(src_op=self.fc1_op,
                                                     name="fc",
                                                     n_dims=[20, dst_dim],
                                                     dropout_rate=0.4)
        # regression の場合はそのままdstとする
        self.dst_op = self.blocks["fc_linear"].get_dst_op()

        # loss_op: 普通の二乗誤差
        const_100 = tf.constant(100.0)
        sq_err = tf.square(self.dst_op*const_100 - self.label_op*const_100)
        self.loss_op = tf.reduce_mean(sq_err)
        # self.dst_op = tf.Print(self.dst_op, [self.dst_op],
        #                        "dst_op_end: ", summarize=100)
        # self.label_op = tf.Print(self.label_op, [self.label_op],
        #                          "label_op_end: ", summarize=100)
        # sq_err = tf.Print(sq_err, [sq_err], "sq_err: ", summarize=100)
        # self.loss_op = tf.Print(self.loss_op, [self.loss_op],
        #                         "loss_op: ", summarize=100)

        # accuracy_op
        th = tf.constant(correct_dis_th**2, dtype=tf.float32)
        correct = tf.less(sq_err, th)
        self.acc_op = tf.reduce_mean(tf.cast(correct, tf.float32))
