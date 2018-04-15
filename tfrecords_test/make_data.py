# _*_ coding: utf-8 _*_

import tensorflow as tf
import numpy as np

from prepare_data import get_img_data

# tfrecords 支持的数据类型
# tf.train.Feature(int64_list = tf.train.Int64List(value=[int_scalar]))
# tf.train.Feature(bytes_list = tf.train.BytesList(value=[array_string_or_byte]))
# tf.train.Feature(bytes_list = tf.train.FloatList(value=[float_scalar]))

# 创建tfrecords文件
file_nums = 2
instance_per_file = 5
dir = "imgs/"

data = get_img_data(dir)  # type(data) list
for i in range(file_nums):
    tfrecords_filename = './tfrecords/train.tfrecords-%.5d-of-%.5d' % (i, file_nums)
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)  # 创建.tfrecord文件

    for j in range(instance_per_file):
        # type(data[i*instance_per_file+j]) numpy.ndarray
        img_raw = np.asarray(data[i*instance_per_file+j]).tostring()

        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[j])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())

    writer.close()
