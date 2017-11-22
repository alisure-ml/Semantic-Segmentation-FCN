import os
import random
import zipfile
import numpy as np
from glob import glob
import scipy.misc as misc
from six.moves import cPickle as pickle

from Tools import Tools


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

    def __init__(self, batch_size, type_number, image_size, image_channel, records_list,
                 image_options=None, shuffle=True, is_test=0):
        self.batch_size = batch_size
        self.type_number = type_number
        self.image_size = image_size
        self.image_channel = image_channel
        self.shuffle = shuffle
        self.is_test = is_test

        self.files = records_list
        if self.is_test != 0:
            self.files = self.files[0: 500]

        self.image_options = image_options
        self._is_resize = True if self.image_options.get("resize", False) and self.image_options["resize"] else False
        self._resize_size = int(self.image_options["resize_size"]) if self._is_resize else 0

        self.images = []
        self.annotations = []

        self.batch_offset = 0
        self.epochs_completed = 0

        self._read_images()

        pass

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

            # Shuffle the data
            if self.shuffle:
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
        return self.images[i * self.batch_size: (i + 1) * self.batch_size], \
               self.annotations[i * self.batch_size: (i + 1) * self.batch_size]

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
