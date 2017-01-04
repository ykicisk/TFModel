#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression


class LRModel(object):
    def __init__(self, feature_info, C, tol=0.01, penalty="l2"):
        u"""warpper"""
        self.C = C
        self.tol = tol
        self.penalty = penalty
        self.sess = None
        self.feature_info = feature_info

    def get_const_placeholders(self, batch_size, train_flag=False):
        u"""学習データとは別にconst値のplaceholderがあれば返す.
        例) RNNの初期state, keep_prob等"""
        ret = {}
        for block in self.blocks.values():
            ret.update(block.get_const_placeholders(batch_size, train_flag))
        return ret

    def init(self):
        if self.sess is None:
            init_op = [tf.initialize_all_variables(),
                       tf.initialize_local_variables()]
            self.sess = tf.Session()
            self.sess.run(init_op)

    def train(self, optimizer, train_batch, validate_batch,
              max_iteration, validate_iteration,
              validate_interval, save_interval,
              model_path_fmt, iter=0):

        self.init()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        # バッチで学習データを取得する
        print "=== load batch ==="
        features = []
        labels = []
        parts = ["daily", "weekly", "monthly"]
        try:
            # ミニバッチ処理ループ
            while iter < max_iteration and not coord.should_stop():
                # self.logger.info("iter: %d", iter)
                batch_dict = self.sess.run(train_batch)
                # print batch_dict.keys()
                # print [v.shape for v in batch_dict.values()]
                bsize = batch_dict.values()[0].shape[0]
                feat = np.concatenate([batch_dict[p].reshape(bsize, -1)
                                       for p in parts], axis=1)
                label = batch_dict["label"][:, 0, :].flatten()
                label = (label >= 0) * 1
                features.append(feat)
                labels.append(label)
                iter += 1
                # TODO validate model?
                # TODO save model?
        except tf.errors.OutOfRangeError:
            print " == OutOfRangeError (WHY???) == "
        finally:
            print " == load end == "
            coord.request_stop()
            coord.join(threads)

        print "=== train LRModel ==="
        X = np.concatenate(features, axis=0)
        y = np.concatenate(labels, axis=0)
        print "X.shape:", X.shape
        print "y.shape:", y.shape
        model = LogisticRegression(C=self.C, penalty=self.penalty,
                                   tol=self.tol)
        model.fit(X, y)
        print "=== coef ==="
        feature_names = [self.feature_info[p]["feature_names"] for p in parts]
        feature_names = sum(feature_names, [])
        fname_coef = zip(feature_names, model.coef_[0])
        print sorted(fname_coef, key=lambda x: x[1])
