import scipy.io as sio
import numpy as np
import tensorflow as tf


class FCN_VGGNet:

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
    def _vgg_net(weights, image):
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

    # 网络
    # keep_prob=0.7
    def vgg_19(self, input_op, **kw):
        centered_image = input_op - self._mean_pixel  # mean

        with tf.variable_scope("inference"):
            # [256, 256] -> [16, 16]
            vgg_net = self._vgg_net(self._weights, centered_image)
            #  [16, 16]
            conv_final_layer = vgg_net["conv5_3"]

            # [8, 8]
            pool5 = tf.nn.max_pool(conv_final_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            W6 = tf.get_variable(name="W6", initializer=tf.truncated_normal([7, 7, 512, 4096], stddev=0.02))
            b6 = tf.get_variable(name="b6", initializer=tf.constant(0.0, shape=[4096]))
            conv6 = tf.nn.bias_add(tf.nn.conv2d(pool5, W6, strides=[1, 1, 1, 1], padding="SAME"), b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=kw["keep_prob"])

            W7 = tf.get_variable(name="W7", initializer=tf.truncated_normal([1, 1, 4096, 4096], stddev=0.02))
            b7 = tf.get_variable(name="b7", initializer=tf.constant(0.0, shape=[4096]))
            conv7 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout6, W7, strides=[1, 1, 1, 1], padding="SAME"), b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=kw["keep_prob"])

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

            shape = tf.shape(input_op)
            deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], self._type_number])

            W_t3 = tf.get_variable(name="W_t3", initializer=tf.truncated_normal([16, 16, self._type_number, deconv_shape2[3].value], stddev=0.02))
            b_t3 = tf.get_variable(name="b_t3", initializer=tf.constant(0.0, shape=[self._type_number]))
            conv_t3 = tf.nn.bias_add(tf.nn.conv2d_transpose(fuse_2, W_t3, deconv_shape3, strides=[1, 8, 8, 1], padding="SAME"), b_t3)

            logits = conv_t3
            prediction = tf.expand_dims(tf.argmax(logits, dimension=3, name="prediction"), dim=3)

        return logits, prediction

    # 改变最后一层反卷积的步长
    # keep_prob=0.7
    def vgg_19_fcn_4(self, input_op, **kw):
        centered_image = input_op - self._mean_pixel  # mean

        with tf.variable_scope("inference"):
            # [256, 256] -> [16, 16]
            vgg_net = self._vgg_net(self._weights, centered_image)
            #  [16, 16]
            conv_final_layer = vgg_net["conv5_3"]

            # [8, 8]
            pool5 = tf.nn.max_pool(conv_final_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

            W6 = tf.get_variable(name="W6", initializer=tf.truncated_normal([7, 7, 512, 4096], stddev=0.02))
            b6 = tf.get_variable(name="b6", initializer=tf.constant(0.0, shape=[4096]))
            conv6 = tf.nn.bias_add(tf.nn.conv2d(pool5, W6, strides=[1, 1, 1, 1], padding="SAME"), b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=kw["keep_prob"])

            W7 = tf.get_variable(name="W7", initializer=tf.truncated_normal([1, 1, 4096, 4096], stddev=0.02))
            b7 = tf.get_variable(name="b7", initializer=tf.constant(0.0, shape=[4096]))
            conv7 = tf.nn.bias_add(tf.nn.conv2d(relu_dropout6, W7, strides=[1, 1, 1, 1], padding="SAME"), b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=kw["keep_prob"])

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

            deconv_shape3 = vgg_net["pool2"].get_shape()
            W_t3 = tf.get_variable(name="W_t3", initializer=tf.truncated_normal([4, 4, deconv_shape3[3].value, deconv_shape2[3].value], stddev=0.02))
            b_t3 = tf.get_variable(name="b_t3", initializer=tf.constant(0.0, shape=[deconv_shape3[3].value]))
            conv_t3 = tf.nn.bias_add(tf.nn.conv2d_transpose(fuse_2, W_t3, tf.shape(vgg_net["pool2"]), strides=[1, 2, 2, 1], padding="SAME"), b_t3)
            fuse_3 = tf.add(conv_t3, vgg_net["pool2"], name="fuse_3")

            shape = tf.shape(input_op)
            deconv_shape4 = tf.stack([shape[0], shape[1], shape[2], self._type_number])

            W_t4 = tf.get_variable(name="W_t4", initializer=tf.truncated_normal([8, 8, self._type_number, deconv_shape3[3].value], stddev=0.02))
            b_t4 = tf.get_variable(name="b_t4", initializer=tf.constant(0.0, shape=[self._type_number]))
            conv_t4 = tf.nn.bias_add(tf.nn.conv2d_transpose(fuse_3, W_t4, deconv_shape4, strides=[1, 4, 4, 1], padding="SAME"), b_t4)

            logits = conv_t4
            prediction = tf.expand_dims(tf.argmax(logits, dimension=3, name="prediction"), dim=3)

        return logits, prediction

    pass
