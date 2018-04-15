import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 读取tfrecords文件
# --------------hyperParams--------------------------
batch_size = 2
capacity = 1000 + 3*batch_size
train_rounds = 5
num_epochs = 30
img_h = 333
img_w = 500
# ---------------------------------------------------

tfrecord_files = tf.train.match_filenames_once('./tfrecords/train.tfrecords-*')
queue = tf.train.string_input_producer(tfrecord_files, num_epochs=num_epochs, shuffle=True, capacity=10)

reader = tf.TFRecordReader()
# 从文件中读出一个队列， 也可以使用read_uo_to函数一次性读取多个样例
_, serialized_example = reader.read(queue)

# 读取多个对应tf.parse_example()
# 读取单个对应tf.parse_single_example()

features = tf.parse_single_example(
    serialized_example, features={
        'label': tf.FixedLenFeature([], tf.int64),
        'img_raw': tf.FixedLenFeature([], tf.string),
    }
)


image = tf.decode_raw(features['img_raw'], tf.uint8)
# image_shape = tf.stack([img_h, img_w, 3])
image = tf.reshape(image, [img_h, img_w, 3])
label = tf.cast(features['label'], tf.int64)


# tf.train.shuffle_batch()
to_train_batch, to_label_batch = tf.train.shuffle_batch(
    [image, label], batch_size=batch_size, capacity=capacity,
    allow_smaller_final_batch=True, num_threads=1, min_after_dequeue=1
)


with tf.Session() as sess:
    sess.run(
        tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    )

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    for i in range(train_rounds):
        train_batch, label_batch = sess.run([to_train_batch, to_label_batch])
        plt.subplot(121)
        plt.imshow(train_batch[0])
        plt.subplot(122)
        plt.imshow(train_batch[1])
        plt.show()
    coord.request_stop()
    coord.join(threads)

print('finish all')
