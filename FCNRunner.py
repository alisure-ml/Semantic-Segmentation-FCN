import os
import argparse
import numpy as np
import tensorflow as tf
import scipy.misc as misc

from Tools import Tools
from DataLoader import PreData, Data
from NetVGG import FCN_VGGNet


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
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(self.label, axis=[3]), logits=self.logits)
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
    def train(self, epochs, loss_freq, test_freq, valid_freq, valid_number, save_freq, result_path):
        with self.supervisor.managed_session(config=self.config) as sess:
            for epoch in range(epochs):
                loss = None
                while self._train_data.epochs_completed == epoch:
                    # stop
                    if self.supervisor.should_stop():
                        break
                    # train
                    x, labels = self._train_data.next_batch()
                    loss, _ = sess.run(fetches=[self.loss, self.train_op], feed_dict={self.image: x, self.label: labels})
                    pass
                if epoch % loss_freq == 0:
                    Tools.print_info("{} loss {}".format(epoch, loss))
                if epoch % valid_freq == 0:
                    self._valid(sess, valid_number, epoch, result_path)
                if epoch % test_freq == 0:
                    self._test(sess, info="test")
                if epoch % save_freq == 0:
                    self.supervisor.saver.save(sess, os.path.join(self._model_path, "model_{}".format(epoch)))
                    pass
                pass
        pass

    # 验证网络
    def _valid(self, sess, valid_number, epoch, result_path):
        Tools.print_info("{} save result".format(epoch))
        for i in range(valid_number):
            images, labels = self._valid_data.get_batch_i(i)
            prediction = sess.run(fetches=self.prediction, feed_dict={self.image: images})
            valid = np.squeeze(labels, axis=3)
            predict = np.squeeze(prediction, axis=3)
            for itr in range(len(images)):
                old_file = os.path.join(result_path, "{}-{}-old.png".format(i, itr))
                if not os.path.exists(old_file):
                    misc.imsave(old_file, images[itr].astype(np.uint8))

                val_file = os.path.join(result_path, "{}-{}-val.png".format(i, itr))
                if not os.path.exists(val_file):
                    misc.imsave(val_file, valid[itr].astype(np.uint8))

                pre_file = os.path.join(result_path, "{}-{}-{}-pre.png".format(i, itr, epoch))
                misc.imsave(pre_file, predict[itr].astype(np.uint8))
                pass
        pass

    def valid(self, valid_number, info, result_path):
        with self.supervisor.managed_session() as sess:
            self._valid(sess, valid_number, info, result_path)
        pass

    # 测试网络
    def test(self, info):
        with self.supervisor.managed_session() as sess:
            self._test(sess, info)
        pass

    def _test(self, sess, info):
        test_count = 0
        test_correct = 0.0

        now_epochs = self._valid_data.epochs_completed
        while self._valid_data.epochs_completed == now_epochs:
            images, labels = self._valid_data.next_batch()
            prediction = sess.run(fetches=self.prediction, feed_dict={self.image: images})
            labels = np.squeeze(labels, axis=3)
            predicts = np.squeeze(prediction, axis=3)

            test_count += 1
            ok_count = 0
            for index in range(len(images)):
                now_label = labels[index].astype(np.uint8)
                now_predict = predicts[index].astype(np.uint8)
                ok_count += np.sum(np.equal(now_label, now_predict))
                pass
            test_correct += ok_count / (len(images) * len(images[0]) * len(images[0][0]))
            pass
        test_correct /= test_count

        Tools.print_info("------------------------------------")
        Tools.print_info(" test correct is {}".format(test_correct))
        Tools.print_info("------------------------------------")
        pass

    pass

if __name__ == '__main__':

    # argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, default="fcn_vgg_19", help="name")
    parser.add_argument("-epochs", type=int, default=100, help="train epoch number")
    parser.add_argument("-batch_size", type=int, default=8, help="batch size")
    parser.add_argument("-type_number", type=int, default=151, help="type number")
    parser.add_argument("-image_size", type=int, default=224, help="image size")
    parser.add_argument("-image_channel", type=int, default=3, help="image channel")
    parser.add_argument("-keep_prob", type=float, default=0.7, help="keep prob")
    parser.add_argument("-valid_number", type=int, default=3, help="valid number")
    parser.add_argument("-is_test", type=int, default=1, help="is test")
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
                          records_list=train_records, image_options=image_options, shuffle=True, is_test=args.is_test)
    now_valid_data = Data(batch_size=args.batch_size, type_number=args.type_number,
                          image_size=args.image_size, image_channel=args.image_channel,
                          records_list=valid_records, image_options=image_options, shuffle=False, is_test=args.is_test)

    # net
    now_net = FCN_VGGNet(args.type_number, args.image_size, args.image_channel, args.batch_size)

    # run
    runner = Runner(train_data=now_train_data, valid_data=now_valid_data, fcn_net=now_net.vgg_19,
                    model_path="model/{}".format(args.name), learning_rate=0.0001, keep_prob=args.keep_prob)
    runner.train(epochs=args.epochs, loss_freq=1, test_freq=1, valid_freq=1, valid_number=args.valid_number,
                 save_freq=1, result_path=Tools.new_dir(os.path.join("result", args.name)))
    runner.valid(valid_number=args.valid_number, info="valid",
                 result_path=Tools.new_dir(os.path.join("result", args.name)))
    runner.test(info="test")

    pass
