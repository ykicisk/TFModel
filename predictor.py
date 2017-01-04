#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import os
import yaml
import json
import argparse as ap
from collections import defaultdict
import copy
import logging

import tensorflow as tf
import pandas as pd

import TFModel


def parse_timeseries_data(data, conf):
    u"""pfをパースして、feature_group->np.arrayの辞書に変換する。
    feature_groupはcolumn名からパース： daily:close_rate1 なら dailyがfeature_group
    conf: {train_term: "20160101:20160630", validate_term, test_term,}
    return: *_data = {feature_group: np.array, ...},
            feature_names = {feature_group: [feat1, feat2, ...]}
    """
    assert(type(data) == pd.Panel)
    # sort index by date
    data = data.sort_index(axis=1)

    # fgroup_map: feature_group -> [fname1, fname2, ...]
    fgroup_map = defaultdict(list)
    for fname in data.ix[0].columns:
        # fg = "{fg}:0".format(fg=fname.split(":", 1)[0])
        fg_name = fname.split(":", 1)
        if len(fg_name) < 2:
            continue
        fg, _ = fg_name
        # fg = fname.split(":", 1)[0]
        fgroup_map[fg].append(fname)

    # split data
    def parse_data(data, term_dump):
        start_dump, end_dump = term_dump.split(":")
        start_pdts = pd.Timestamp(start_dump) if start_dump else pd.Timestamp("19900101")
        end_pdts = pd.Timestamp(end_dump) if end_dump else pd.Timestamp("20501231")
        return data.ix[:, pd.IndexSlice[start_pdts:end_pdts], :]
    train_data = parse_data(data, conf["train_term"])
    validate_data = parse_data(data, conf["validate_term"])
    test_data = parse_data(data, conf["test_term"])
    return train_data, validate_data, test_data, fgroup_map


def get_file_paths(conf):
    filepaths = glob.glob("{root}/*/*/*.tfrecords"
                          .format(root=conf["src"]))
    terms = {}
    for part in ["train", "validate", "test"]:
        start_dump, end_dump = conf["{p}_term".format(p=part)].split(":")
        start_pdts = pd.Timestamp(start_dump) \
            if start_dump else pd.Timestamp("19900101")
        end_pdts = pd.Timestamp(end_dump) \
            if end_dump else pd.Timestamp("20501231")
        terms[part] = {"start": start_pdts, "end": end_pdts}

    filepath_dict = {"train": [], "validate": [], "test": []}
    for p in filepaths:
        pdts_elems = os.path.splitext(p)[0].rsplit("/", 3)
        pdts = pd.Timestamp("".join(pdts_elems[-3:]))
        for part in ["train", "validate", "test"]:
            if terms[part]["start"] <= pdts and terms[part]["end"] >= pdts:
                filepath_dict[part].append(p)
                break

    return filepath_dict


def get_batch(feature_info, filepaths, batch_size, num_threads):
    # print filepaths
    file_queue = tf.train.string_input_producer(filepaths, shuffle=True)
    reader = tf.TFRecordReader()
    # key => filepath:num
    key, serialized_data = reader.read(file_queue)

    # Parse TFRecord setting

    features_dict = {fgroup: tf.FixedLenFeature(info["shape"], tf.float32)
                     for fgroup, info in feature_info.items()}
    # print features_dict
    features = tf.parse_single_example(serialized_data,
                                       features=features_dict)

    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size

    min_after_dequeue = batch_size * 100
    capacity = min_after_dequeue + 3 * batch_size
    batch = tf.train.shuffle_batch(
        features,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads,
        allow_smaller_final_batch=True
    )

    return batch


def main(config_path, mode, iter):
    u"""main func"""
    conf = yaml.load(open(config_path))

    # logging setting
    log_level_dict = {
        "debug": logging.DEBUG, "info": logging.info,
        "warning": logging.warning, "error": logging.error,
        "critical": logging.critical
    }
    # logging.basicConfig(format="%(levelname)s: [%(message)s]",
    logging.basicConfig(level=log_level_dict[conf["log_level"]])

    print "-- get {train,validate,test} => filepaths --"
    filepath_dict = get_file_paths(conf)
    print "train files:", len(filepath_dict["train"])
    print "varidate files:", len(filepath_dict["validate"])
    print "test files:", len(filepath_dict["test"])
    print "--------------------------------------------"
    feature_info = json.load(open(os.path.join(conf["src"], "info.json")))

    model_class = getattr(TFModel, conf["model"]["name"])

    model_params = copy.deepcopy(conf["model"]["params"])
    model_params["feature_info"] = feature_info
    model_params["board_logdir"] = conf["board_logdir"]
    model = model_class(**model_params)

    # get mini batch config
    batch_size = conf["batch_size"]
    n_thread = conf["batch_threads"]

    if mode == "train":
        opti_class = getattr(tf.train, conf["optimizer"]["name"])
        optimizer = opti_class(**conf["optimizer"]["params"])

        train_batch = get_batch(feature_info, filepath_dict["train"],
                                batch_size, n_thread)
        validate_batch = get_batch(feature_info, filepath_dict["validate"],
                                   batch_size, n_thread)

        # learning!!
        train_params = copy.deepcopy(conf["train_params"])
        if iter > 0:
            model.load(train_params["model_path_fmt"].format(iter=iter))
        train_params["optimizer"] = optimizer
        train_params["train_batch"] = train_batch
        train_params["validate_batch"] = validate_batch
        train_params["iter"] = iter
        model.train(**train_params)

    elif mode == "test":
        test_params = conf["test_params"]
        file_fmt = test_params["model_path_fmt"]
        if iter == 0:
            # get max_iteration
            prefix, surfix = file_fmt.split("{iter}")
            candidates = glob.glob(file_fmt.format(iter="*"))
            for c in candidates:
                if c.startswith(prefix) and c.endswith(surfix):
                    now_iter = int(c[len(prefix):-len(surfix)])
                    iter = max(iter, now_iter)
        model.load(file_fmt.format(iter=iter))

        test_batch = get_batch(feature_info, filepath_dict["test"],
                               batch_size, n_thread)
        max_iteration = test_params["iteration"]
        _, _, loss, accuracy = model.test(test_batch, max_iteration)
        # TODO 結果ファイルを作る
        print "loss:{loss}\taccuracy:{acc}".format(loss=loss,
                                                   acc=accuracy)


if __name__ == "__main__":
    description = """predict stock price

-- config yaml format --
...coming soon...
"""

    class Formatter(ap.ArgumentDefaultsHelpFormatter,
                    ap.RawDescriptionHelpFormatter):
        pass
    parser = ap.ArgumentParser(description=description,
                               formatter_class=Formatter)
    parser.add_argument("mode", choices=["train", "test"], help="run mode")
    parser.add_argument("conf", help="config yaml path")
    parser.add_argument("--iter", default=0, type=int,
                        help="train: start fine tuning from target model,"
                        "test: test target model")
    args = parser.parse_args()
    main(args.conf, args.mode, args.iter)
