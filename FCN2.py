# encoding:UTF-8
from __future__ import print_function
import tensorflow as tf
import numpy as np
import scipy.io as sio
import datetime
from Data import Data
import scipy.misc as misc
import os

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("result_dir", "result", "path to result directory")
tf.flags.DEFINE_string("iter", 1000, "iter to train")


class FCN_VGG_Net:

    def __init__(self, type_number, image_size, image_channel, batch_size,
                 model_file="data/imagenet-vgg-verydeep-19.mat"):
        self._type_number = type_number
        self._image_size = image_size
        self._image_channel = image_channel
        self._batch_size = batch_size

        self._model_data = sio.loadmat(model_file)
        self._weights = np.squeeze(self._model_data['layers'])
        self._mean_pixel = np.mean(self._model_data['normalization'][0][0][0], axis=(0, 1))
        pass

    @staticmethod
    def vgg_net(weights, image):
        layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                  'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                  'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
                  'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
                  'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4']

        net = {}

        current = image
        for i, name in enumerate(layers):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                w = np.transpose(kernels, (1, 0, 2, 3))
                b = bias.reshape(-1)
                kernels = tf.get_variable(name=name + "_w", initializer=tf.constant_initializer(w, dtype=tf.float32), shape=w.shape)
                bias = tf.get_variable(name=name + "_b", initializer=tf.constant_initializer(b, dtype=tf.float32), shape=b.shape)
                current = tf.nn.bias_add(tf.nn.conv2d(current, kernels, strides=[1, 1, 1, 1], padding="SAME"), bias)
            elif kind == 'relu':
                current = tf.nn.relu(current, name=name)
            elif kind == 'pool':
                current = tf.nn.avg_pool(current, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            net[name] = current

        return net

    # Semantic segmentation network definition
    def inference(self, image, keep_prob):
        centered_image = image - self._mean_pixel  # mean
        with tf.variable_scope("inference"):
            vgg_net = self.vgg_net(self._weights, centered_image)
            conv_final_layer = vgg_net["conv5_3"]

            pool5 = tf.nn.max_pool(conv_final_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            W6 = tf.get_variable(name="W6", initializer=tf.truncated_normal([7, 7, 512, 4096], stddev=0.02))
            b6 = tf.get_variable(name="b6", initializer=tf.constant(0.0, shape=[4096]))
            conv6 = tf.nn.bias_add(tf.nn.conv2d(pool5, W6, strides=[1, 1, 1, 1], padding="SAME"), b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

            W7 = tf.get_variable(name="W7", initializer=tf.truncated_normal([1, 1, 4096, 4096], stddev=0.02))
            b7 = tf.get_variable(name="b7", initializer=tf.constant(0.0, shape=[4096]))
            conv7 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout6, W7, strides=[1, 1, 1, 1], padding="SAME"), b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

            W8 = tf.get_variable(name="W8", initializer=tf.truncated_normal([1, 1, 4096, self._type_number], stddev=0.02))
            b8 = tf.get_variable(name="b8", initializer=tf.constant(0.0, shape=[self._type_number]))
            conv8 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout7, W8, strides=[1, 1, 1, 1], padding="SAME"), b8)

            # now to upscale to actual image size
            deconv_shape1 = vgg_net["pool4"].get_shape()
            W_t1 = tf.get_variable(name="W_t1", initializer=tf.truncated_normal([4, 4, deconv_shape1[3].value, self._type_number], stddev=0.02))
            b_t1 = tf.get_variable(name="b_t1", initializer=tf.constant(0.0, shape=[deconv_shape1[3].value]))
            conv_t1 = tf.nn.bias_add(tf.nn.conv2d_transpose(conv8, W_t1, tf.shape(vgg_net["pool4"]), strides=[1, 2, 2, 1], padding="SAME"), b_t1)
            fuse_1 = tf.add(conv_t1, vgg_net["pool4"], name="fuse_1")

            deconv_shape2 = vgg_net["pool3"].get_shape()
            W_t2 = tf.get_variable(name="W_t2", initializer=tf.truncated_normal([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], stddev=0.02))
            b_t2 = tf.get_variable(name="b_t2", initializer=tf.constant(0.0, shape=[deconv_shape2[3].value]))
            conv_t2 = tf.nn.bias_add(tf.nn.conv2d_transpose(fuse_1, W_t2, tf.shape(vgg_net["pool3"]), strides=[1, 2, 2, 1], padding="SAME"), b_t2)
            fuse_2 = tf.add(conv_t2, vgg_net["pool3"], name="fuse_2")

            shape = tf.shape(image)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self._type_number])

            W_t3 = tf.get_variable(name="W_t3", initializer=tf.truncated_normal([16, 16, self._type_number, deconv_shape2[3].value], stddev=0.02))
            b_t3 = tf.get_variable(name="b_t3", initializer=tf.constant(0.0, shape=[self._type_number]))
            conv_t3 = tf.nn.bias_add(tf.nn.conv2d_transpose(fuse_2, W_t3, deconv_shape3, strides=[1, 8, 8, 1], padding="SAME"), b_t3)

            annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        return tf.expand_dims(annotation_pred, dim=3), conv_t3

    pass


def main():
    image_size = 224
    batch_size = 32

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    image = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 3], name="input_image")
    annotation = tf.placeholder(tf.int32, shape=[None, image_size, image_size, 1], name="annotation")

    fcn_net = FCN_VGG_Net(151, image_size, 3, batch_size)
    pred_annotation, logits = fcn_net.inference(image, keep_prob)
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.squeeze(annotation, squeeze_dims=[3]), name="entropy")))
    train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.result_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    print("Setting up image reader...")
    train_records, valid_records = Data.read_scene_image()
    print(len(train_records))
    print(len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': image_size}
    train_dataset_reader = Data(train_records, image_options)
    validation_dataset_reader = Data(valid_records, image_options)

    for iter in range(int(FLAGS.iter)):
        train_images, train_annotations = train_dataset_reader.next_batch(batch_size)
        feed_dict = {image: train_images, annotation: train_annotations, keep_prob: 0.85}
        sess.run(train_op, feed_dict=feed_dict)

        if iter % 10 == 0:
            train_loss = sess.run(loss, feed_dict=feed_dict)
            print("Step: %d, Train_loss:%g" % (iter, train_loss))

        if iter % 500 == 0:
            valid_images, valid_annotations = validation_dataset_reader.next_batch(batch_size)
            valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations, keep_prob: 1.0})
            print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
            saver.save(sess, os.path.join(FLAGS.result_dir + "model.ckpt"), iter)
        pass

    # valid
    valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
    pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations, keep_prob: 1.0})
    valid_annotations = np.squeeze(valid_annotations, axis=3)
    pred = np.squeeze(pred, axis=3)
    for itr in range(FLAGS.batch_size):
        misc.imsave(os.path.join(FLAGS.result_dir, "inp_" + str(5+itr) + ".png"), valid_images[itr].astype(np.uint8))
        misc.imsave(os.path.join(FLAGS.result_dir, "gt_" + str(5+itr) + ".png"), valid_annotations[itr].astype(np.uint8))
        misc.imsave(os.path.join(FLAGS.result_dir, "pred_" + str(5+itr) + ".png"), pred[itr].astype(np.uint8))
        print("Saved image: %d" % itr)
        pass

    pass


if __name__ == "__main__":
    main()
