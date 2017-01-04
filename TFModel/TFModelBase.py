#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
from abc import ABCMeta, abstractmethod
import numpy as np
import tensorflow as tf
import logging


class TFModelBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, feature_info, board_logdir):
        u"""ここの__init__で以下のものを定義する
        self.loss_op: optimizerで最小化するlossのオペレータ
        self.acc_op: 精度のオペレータ（学習・テスト時に利用）
        """
        self.sess = tf.Session()
        self.board_logdir = board_logdir
        self.logger = logging.getLogger(type(self).__name__)

        self.feed_ph = {}
        for fgroup, info in feature_info.items():
            self.feed_ph[fgroup] = tf.placeholder(tf.float32,
                                                  [None] + info["shape"],
                                                  name=fgroup)

    @abstractmethod
    def get_const_placeholders(self, batch_size, train_flag=False):
        u"""学習データとは別にconst値のplaceholderがあれば返す.
        例) RNNの初期state, keep_prob等"""
        return {}

    def init(self):
        self.merge_summary_op = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.board_logdir,
                                                     self.sess.graph)
        init_op = [tf.initialize_all_variables(),
                   tf.initialize_local_variables()]
        self.sess.run(init_op)

    def train_pre_processing(self, train_dict):
        pass

    def print_test_summary(self, dst, label):
        pass

    def train(self, optimizer, train_batch, validate_batch,
              max_iteration, validate_iteration,
              validate_interval, save_interval,
              model_path_fmt, iter=0):

        # create model destination directory
        model_dir = os.path.dirname(model_path_fmt)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # learning
        self.init()
        optimize_op = optimizer.minimize(self.loss_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        print "=== start train ==="
        try:
            # ミニバッチ処理ループ
            while iter < max_iteration and not coord.should_stop():
                # self.logger.info("iter: %d", iter)
                batch_dict = self.sess.run(train_batch)
                # self.logger.debug("batch_dict.keys(): %s", batch_dict.keys())
                bsize = batch_dict[batch_dict.keys()[0]].shape[0]
                # self.logger.debug("bsize: %d", bsize)
                ex_dict = self.get_const_placeholders(bsize,
                                                      train_flag=True)
                train_dict = {}
                # why need ":0" ?
                # see: https://github.com/tensorflow/tensorflow/issues/3378
                for k, v in batch_dict.items():
                    train_dict["{key}:0".format(key=k)] = v
                train_dict.update(ex_dict)
                # self.logger.debug("train_dict.keys(): %s", train_dict.keys())
                self.train_pre_processing(train_dict)
                self.sess.run(optimize_op, feed_dict=train_dict)

                iter += 1

                # validate model
                if iter % validate_interval == 0:
                    print "iter:", iter
                    dst, label, loss, acc, summary\
                        = self.test(validate_batch, validate_iteration)
                    self.print_test_summary(dst, label)
                    self.summary_writer.add_summary(summary, iter)
                    print "loss:{loss}\taccuracy:{acc}".format(loss=loss,
                                                               acc=acc)

                # save model
                if iter % save_interval == 0:
                    model_path = model_path_fmt.format(iter=iter)
                    self.save(model_path)

        except tf.errors.OutOfRangeError:
            print " == OutOfRangeError (WHY???) == "
        finally:
            print " == train finally == "
            coord.request_stop()
            coord.join(threads)

    def test(self, test_batch, max_iteration):
        u"""テストを行う
        return: result, label, loss, accuracy"""
        n_sample = 0
        sum_loss = 0.0
        sum_acc = 0.0
        dst_list = []

        iter = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        labels = []
        try:
            # ミニバッチ処理ループ
            while iter < max_iteration and not coord.should_stop():
                batch_dict = self.sess.run(test_batch)
                bsize = batch_dict[batch_dict.keys()[0]].shape[0]
                ex_dict = self.get_const_placeholders(bsize,
                                                      train_flag=True)
                test_dict = {}
                # why need ":0" ?
                # see: https://github.com/tensorflow/tensorflow/issues/3378
                for k, v in batch_dict.items():
                    test_dict["{key}:0".format(key=k)] = v
                test_dict.update(ex_dict)
                labels.append(test_dict["label:0"])
                sum_loss += bsize*self.sess.run(self.loss_op,
                                                feed_dict=test_dict)
                sum_acc += bsize*self.sess.run(self.acc_op,
                                               feed_dict=test_dict)
                dst_list.append(self.sess.run(self.dst_op,
                                              feed_dict=test_dict))
                n_sample += bsize

                iter += 1
        except tf.errors.OutOfRangeError:
            print " == OutOfRangeError (WHY???) == "
        finally:
            # print " == finally == "
            coord.request_stop()
            coord.join(threads)
        dst = np.vstack(dst_list)
        loss = sum_loss/n_sample
        acc = sum_acc/n_sample

        # Tensorboard summary
        summary = self.sess.run(self.merge_summary_op)

        return dst, np.concatenate(labels), loss, acc, summary

    def save(self, model_path):
        u"""学習済みモデルファイルを保存する"""
        print "save model:", model_path
        if self.sess is None:
            raise Exception("No session exception")
        saver = tf.train.Saver()
        saver.save(self.sess, model_path)

    def load(self, model_path):
        u"""学習済みモデルファイルをロードする"""
        print "load model:", model_path
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, model_path)


