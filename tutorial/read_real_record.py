#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import argparse as ap
import tensorflow as tf


def main(src_glob):
    print "get filepaths"
    filepaths = glob.glob(src_glob)

    print "reader test!!"
    num_epochs = None
    file_queue = tf.train.string_input_producer(filepaths,
                                                num_epochs=num_epochs,
                                                shuffle=True)

    reader = tf.TFRecordReader()
    key, selialized_data = reader.read(file_queue)  # key => filepath:num
    print type(key), type(selialized_data)

    # dim
    features_dict = {
        "daily": tf.FixedLenFeature([5, 56], tf.float32),
        "weekly": tf.FixedLenFeature([5, 58], tf.float32),
        "monthly": tf.FixedLenFeature([5,58], tf.float32),
        "label": tf.FixedLenFeature([5,1], tf.float32)
    }
    features = tf.parse_single_example(selialized_data,
                                       features=features_dict)

    batch_size = 10
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    num_threads = 1
    min_after_dequeue = batch_size * 100
    capacity = min_after_dequeue + 3 * batch_size
    features_batch = tf.train.shuffle_batch(
        features,
        batch_size=batch_size,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        num_threads=num_threads,
        allow_smaller_final_batch=True
    )

    print type(features_batch)
    for fgroup, batch in features_batch.items():
        print fgroup, type(batch)
    init_op = [tf.initialize_all_variables(),
               tf.initialize_local_variables()]

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            # ミニバッチ処理ループ
            while not coord.should_stop():
                key_ = sess.run(key)
                print key_
                sample = sess.run(features_batch.values())
                print sample
                # 学習等の処理
        except tf.errors.OutOfRangeError:
            print "epoch end"
        finally:
            print "===finally==="
            coord.request_stop()
            coord.join(threads)

if __name__ == "__main__":
    description = """start"""

    class Formatter(ap.ArgumentDefaultsHelpFormatter,
                    ap.RawDescriptionHelpFormatter):
        pass
    parser = ap.ArgumentParser(description=description,
                               formatter_class=Formatter)
    parser.add_argument("src", help="input tfrecord path (glob format)")
    args = parser.parse_args()
    main(args.src)
