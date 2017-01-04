#!/usr/bin/env python
# -*- coding:utf-8 -*-
import argparse as ap
import numpy as np
import tensorflow as tf


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def main(dst_path, base_idx):
    with tf.python_io.TFRecordWriter(dst_path) as writer:
        a = np.array([
                        [[1,2,3],[3.2,4,5]],
                        [[1.4,2.2,3.3],[3.2,4.2,5.5]],
                        ])
        print a.shape
        for i in range(1000):
            example = tf.train.Example(features=tf.train.Features(feature={
                'int_key': _int64_feature(base_idx),
                'float': _floats_feature([i, 1.234]),
                'array': _floats_feature(a.flatten().tolist())
            }))
            writer.write(example.SerializeToString())

if __name__ == "__main__":
    description = """start"""

    class Formatter(ap.ArgumentDefaultsHelpFormatter,
                    ap.RawDescriptionHelpFormatter):
        pass
    parser = ap.ArgumentParser(description=description,
                               formatter_class=Formatter)
    parser.add_argument("dst", help="output tfrecord path")
    parser.add_argument("base", type=int, help="int base")
    args = parser.parse_args()
    main(args.dst, args.base)
