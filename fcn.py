# encoding: UTF-8
import os
import time
import random
import zipfile
import argparse
import scipy.io as sio
import numpy as np
from glob import glob
import tensorflow as tf
import scipy.misc as misc
from six.moves import cPickle as pickle


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    pass


class PreData:

    @staticmethod
    def get_scene_image(data_dir="data", data_zip="ADEChallengeData2016.zip"):
        pickle_path = os.path.join(data_dir, "scene_image.pickle")
        if not os.path.exists(pickle_path):
            new_data_path = os.path.join(data_dir, data_zip)
            new_data_dir = new_data_path.split(".")[0]
            if not os.path.exists(new_data_dir):
                with zipfile.ZipFile(new_data_path) as zf:
                    zf.extractall(data_dir)
            result = PreData._create_image_lists(new_data_dir)
            with open(pickle_path, 'wb') as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
            pass
        return pickle_path

    @staticmethod
    def _create_image_lists(image_dir):
        if not os.path.exists(image_dir):
            Tools.print_info("Image directory '" + image_dir + "' not found.")
            return None
        directories = ['training', 'validation']
        image_list = {}

        for directory in directories:
            file_list = []
            image_list[directory] = []
            file_glob = os.path.join(image_dir, "images", directory, '*.' + 'jpg')
            file_list.extend(glob(file_glob))

            if not file_list:
                Tools.print_info('No files found')
            else:
                for file_name in file_list:
                    filename = os.path.splitext(file_name.split("/")[-1])[0]
                    annotation_file = os.path.join(image_dir, "annotations", directory, filename + '.png')
                    if os.path.exists(annotation_file):
                        record = {'image': file_name, 'annotation': annotation_file, 'filename': filename}
                        image_list[directory].append(record)
                    else:
                        Tools.print_info("Annotation file not found for %s - Skipping" % filename)
                pass
            random.shuffle(image_list[directory])
            Tools.print_info('No. of %s files: %d' % (directory, len(image_list[directory])))
        return image_list

    @staticmethod
    def main():
        return PreData.get_scene_image()

    pass


class Data:

    def __init__(self, batch_size, type_number, image_size, image_channel, records_list, image_options=None):
        self.batch_size = batch_size
        self.type_number = type_number
        self.image_size = image_size
        self.image_channel = image_channel

        self.files = records_list
        self.image_options = image_options
        self._is_resize = True if self.image_options.get("resize", False) and self.image_options["resize"] else False
        self._resize_size = int(self.image_options["resize_size"]) if self._is_resize else 0

        self.images = []
        self.annotations = []

        self.batch_offset = 0
        self.epochs_completed = 0

        self._read_images()

    def _read_images(self):
        self.images = np.array([self._transform(filename['image'], isImage=True) for filename in self.files])
        self.annotations = np.array([np.expand_dims(self._transform(filename['annotation'], isImage=False), axis=3) for filename in self.files])
        pass

    def _transform(self, filename, isImage):
        image = misc.imread(filename)
        if isImage and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for _ in range(3)])

        if self._is_resize:
            resize_image = misc.imresize(image, [self._resize_size, self._resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def next_batch(self):
        start = self.batch_offset
        self.batch_offset += self.batch_size

        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")

            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]

            # Start next epoch
            start = 0
            self.batch_offset = self.batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]

    def get_batch_i(self, i):
        return self.images[i: i + self.batch_size], self.annotations[i: i + self.batch_size]

    @staticmethod
    def read_scene_image(pickle_path):
        if not os.path.exists(pickle_path):
            raise Exception("{} is not found, please run PreData.get_scene_image()".format(pickle_path))
        with open(pickle_path, 'rb') as f:
            result = pickle.load(f)
            training_records = result['training']
            validation_records = result['validation']
        return training_records, validation_records

    pass


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
    def vgg_16(self, input_op, **kw):
        centered_image = input_op - self._mean_pixel  # mean

        with tf.variable_scope("inference"):
            vgg_net = self._vgg_net(self._weights, centered_image)
            conv_final_layer = vgg_net["conv5_3"]

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

    pass


class Runner:

    def __init__(self, train_data, valid_data, fcn_net, model_path, learning_rate, **kw):
        self._train_data = train_data
        self._valid_data = valid_data
        self._type_number = self._train_data.type_number
        self._image_size = self._train_data.image_size
        self._image_channel = self._train_data.image_channel
        self._batch_size = self._train_data.batch_size
        self._fcn_net = fcn_net
        self._learning_rate = learning_rate
        self._model_path = model_path

        self.graph = tf.Graph()
        # 输入
        self.image, self.label = None, None
        # 网络输出
        self.logits, self.prediction = None, None
        # 损失和训练
        self.loss, self.train_op = None, None

        with self.graph.as_default():
            input_shape = [self._batch_size, self._image_size, self._image_size, self._image_channel]
            self.image = tf.placeholder(shape=input_shape, dtype=tf.float32)
            self.label = tf.placeholder(dtype=tf.int32, shape=[self._batch_size, self._image_size, self._image_size, 1])
            self.logits, self.prediction = self._fcn_net(self.image, **kw)
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.label, squeeze_dims=[3]), logits=self.logits)
            self.loss = tf.reduce_mean(entropy)

            trainable_var = tf.trainable_variables()
            self.train_op = self._train_op(self.loss, trainable_var)
            pass
        self.supervisor = tf.train.Supervisor(graph=self.graph, logdir=self._model_path)
        self.config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        pass

    # train op
    def _train_op(self, loss_val, var_list):
        optimizer = tf.train.AdamOptimizer(self._learning_rate)
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        return optimizer.apply_gradients(grads)

    # 训练网络
    def train(self, epochs, save_model, min_loss, print_loss, test, test_number, save, result_path):
        with self.supervisor.managed_session(config=self.config) as sess:
            epoch = 0
            for epoch in range(epochs):
                # stop
                if self.supervisor.should_stop():
                    break
                # train
                x, labels = self._train_data.next_batch()
                loss, _ = sess.run(fetches=[self.loss, self.train_op], feed_dict={self.image: x, self.label: labels})
                if epoch % print_loss == 0:
                    Tools.print_info("{}: loss {}".format(epoch, loss))
                if loss < min_loss:
                    break
                if epoch % test == 0:
                    self.valid(sess, test_number, epoch, result_path)
                if epoch % save == 0:
                    self.supervisor.saver.save(sess, os.path.join(save_model, "model_{}".format(epoch)))
                pass
            Tools.print_info("{}: train end".format(epoch))
            self.valid(sess, test_number, epoch, result_path)
            Tools.print_info("test end")
        pass

    # 测试网络
    def valid(self, sess, test_number, epoch, result_path):
        Tools.print_info("{} save result".format(epoch))
        for i in range(test_number):
            images, labels = self._valid_data.get_batch_i(i)
            prediction = sess.run(fetches=self.prediction, feed_dict={self.image: images})
            valid = np.squeeze(labels, axis=3)
            predict = np.squeeze(prediction, axis=3)
            for itr in range(len(images)):
                old_file = os.path.join(result_path, "{}-{}-old.png".format(epoch, itr))
                if not os.path.exists(old_file):
                    misc.imsave(old_file, images[itr].astype(np.uint8))

                val_file = os.path.join(result_path, "{}-{}-val.png".format(epoch, itr))
                if not os.path.exists(val_file):
                    misc.imsave(val_file, valid[itr].astype(np.uint8))

                pre_file = os.path.join(result_path, "{}-{}-pre.png".format(epoch, itr))
                misc.imsave(pre_file, predict[itr].astype(np.uint8))
                pass
        pass

    pass

if __name__ == '__main__':

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="fcn_vgg_16", help="name")
    parser.add_argument("-epochs", type=int, default=50000, help="train epoch number")
    parser.add_argument("-batch_size", type=int, default=4, help="batch size")
    parser.add_argument("-type_number", type=int, default=151, help="type number")
    parser.add_argument("-image_size", type=int, default=224, help="image size")
    parser.add_argument("-image_channel", type=int, default=3, help="image channel")
    parser.add_argument("-keep_prob", type=float, default=0.7, help="keep prob")
    args = parser.parse_args()

    # print argument
    output_param = "name={},epochs={},batch_size={},type_num={},size={},channel={},keep_prob={}"
    Tools.print_info(output_param.format(args.name, args.epochs, args.batch_size, args.type_number,
                                         args.image_size, args.image_channel, args.keep_prob))

    # data
    data_path = PreData.main()
    train_records, valid_records = Data.read_scene_image(data_path)
    image_options = {'resize': True, 'resize_size': args.image_size}
    now_train_data = Data(batch_size=args.batch_size, type_number=args.type_number,
                          image_size=args.image_size, image_channel=args.image_channel,
                          records_list=train_records, image_options=image_options)
    now_valid_data = Data(batch_size=args.batch_size, type_number=args.type_number,
                          image_size=args.image_size, image_channel=args.image_channel,
                          records_list=valid_records, image_options=image_options)

    # net
    now_net = FCN_VGGNet(args.type_number, args.image_size, args.image_channel, args.batch_size)

    # run
    runner = Runner(train_data=now_train_data, valid_data=now_valid_data, fcn_net=now_net.vgg_16,
                    model_path="model/{}".format(args.name), learning_rate=0.0001, keep_prob=args.keep_prob)
    runner.train(epochs=args.epochs, save_model=os.path.join("model", args.name),
                 min_loss=1e-4, print_loss=200, test=1000, test_number=5, save=10000,
                 result_path=Tools.new_dir(os.path.join("result", args.name)))

    pass
