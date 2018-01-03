from keras.preprocessing.image import *
import os
import numpy as np
from tqdm import tqdm
import sys
file_path = os.path.dirname(os.path.abspath(__file__))


class MyImageDataGenerator(ImageDataGenerator):
    """docstring for MyImageDataGenerator"""

    def __init__(self, *args, **kwargs):
        super(MyImageDataGenerator, self).__init__(*args, **kwargs)

    def fit_generator(self,
                      generator,
                      steps,
                      save_file='./data/image_channel_mean_std.npy'):
        if os.path.isfile(save_file):
            mean, std = np.load(save_file)
            self.mean = mean
            self.std = std
            return
        x_mean = [0., 0., 0.]
        xsquare_mean = [0., 0., 0.]
        for i in tqdm(range(steps)):
            print i
            #nhwc
            x = next(generator)[0]
            x_mean += 1. / (i + 1) * (
                np.mean(x, axis=(0, self.row_axis, self.col_axis)) - x_mean)
            xsquare_mean += 1. / (i + 1) * (
                np.mean(x**2, axis=(0, self.row_axis,
                                    self.col_axis)) - xsquare_mean)
        self.mean = x_mean
        #var[x] = E[x^2] - (E[x])^2
        self.std = np.sqrt(xsquare_mean - x_mean**2)
        print("data mean:{}, std:{}".format(self.mean, self.std))
        np.save(save_file, (self.mean, self.std))

    def num_samples_per_epoch(self,
                              label_file_path,
                              limit_number=None,
                              balance=False):
        sample_dict = {}
        for l in open(label_file_path, 'r'):
            x, y = l.split()
            if int(y) in sample_dict:
                sample_dict[int(y)].append(x)
            else:
                sample_dict[int(y)] = [
                    x,
                ]
        min_n = sys.maxint
        num_class = len(sample_dict)
        total_num = 0
        for i in sample_dict:
            class_i_num = len(sample_dict[i])
            total_num += class_i_num
            print("class {} has {} samples".format(i, class_i_num))
            min_n = min(min_n, class_i_num)
        if balance:
            total_num = min_n * num_class
            print("after balanced, total sample is {}".format(
                num_class * min_n))
        if limit_number:
            total_num = min(total_num, limit_number)
            print("only predict limit_number of {}".format(total_num))
        num_per_class = int(total_num / num_class)
        #incase that limit_number could not be divided by num_class
        total_num = num_per_class * num_class
        return total_num, sample_dict

    def flow_from_label_file(self,
                             label_file_path,
                             balance=False,
                             nb_output=1,
                             limit_number=None,
                             target_size=(224, 224),
                             batch_size=32,
                             shuffle=True,
                             seed=None,
                             phase='train',
                             is_ergodic_files=False):
        return LabelFileIterator(
            label_file_path,
            image_data_generator=self,
            balance=balance,
            nb_output=nb_output,
            limit_number=limit_number,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=None,
            phase=phase,
            is_ergodic_files=is_ergodic_files)


class LabelFileIterator(Iterator):
    """iterate data from label file"""

    def __init__(self,
                 label_file_path,
                 image_data_generator,
                 balance=False,
                 nb_output=1,
                 limit_number=None,
                 target_size=(299, 299),
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 phase='train',
                 is_ergodic_files=False):
        self.label_file_path = label_file_path
        self.image_data_generator = image_data_generator
        self.balance = balance
        self.target_size = target_size
        self.image_shape = target_size + (3, )
        self.batch_size = batch_size
        self.nb_output = nb_output
        num_per_epoch, sample_dict = self.image_data_generator.num_samples_per_epoch(
            label_file_path, limit_number=limit_number, balance=balance)
        self.num_per_epoch = num_per_epoch
        self.num_class = len(sample_dict)
        self.y_dimension = max(sample_dict) + 1
        self.num_per_class = num_per_epoch / self.num_class
        self.sample_dict = sample_dict
        self.total_epochs_seen = 0
        self.seed = seed
        index = list(range(len(self.splitOpenImage(sample_dict[0])[0])))
        np.random.shuffle(index)
        self.openImageIndex = index
        self.phase = phase
        self.is_ergodic_files = is_ergodic_files
        super(LabelFileIterator, self).__init__(num_per_epoch, batch_size,
                                                shuffle, seed)
        self.broken_recorder = os.path.join(file_path, './brokenImages.log')
        if not os.path.isfile(self.broken_recorder):
            os.mknod(self.broken_recorder)

        # self.x_path_list, self.y_list = [], []
        self.x_path_list, self.y_list = self.gen_data_path_list()

    def steps_per_epoch(self):
        return self.num_per_epoch // self.batch_size

    #get the x:file list and y:lable by random sample
    def gen_data_path_list(self):
        x_path_list, y_list = [], []
        if self.seed:
            np.random.seed(self.seed + self.total_epochs_seen)
        for i in self.sample_dict:
            if self.balance:
                sample = np.random.choice(
                    self.sample_dict[i], self.num_per_class, replace=False)
            else:
                sample = self.sample_dict[i]
            sample_y = [i] * len(sample)
            x_path_list.extend(sample)
            y_list.extend(sample_y)
        x_path_list, y_list = np.array(x_path_list), np.array(
            y_list, dtype='int32')
        if self.shuffle:
            indices = np.arange(len(x_path_list))
            np.random.shuffle(indices)
            x_path_list = x_path_list[indices]
            y_list = y_list[indices]
        print("gen data path list...")
        return x_path_list, y_list

    def gen_ergodic_data_path_list(self):
        x_path_list, y_list = [], []
        if self.seed:
            np.random.seed(self.seed + self.total_epochs_seen)
        for i in self.sample_dict:
            if i == 0:
                sample = self.selectOpenImages(
                    self.sample_dict[i], self.num_per_class, self.balance,
                    self.openImageIndex)
            else:
                if self.balance:
                    sample = np.random.choice(
                        self.sample_dict[i], self.num_per_class, replace=False)
                else:
                    sample = self.sample_dict[i]
            sample_y = [i] * len(sample)
            x_path_list.extend(sample)
            y_list.extend(sample_y)
        x_path_list, y_list = np.array(x_path_list), np.array(
            y_list, dtype='int32')
        if self.shuffle:
            indices = np.arange(len(x_path_list))
            np.random.shuffle(indices)
            x_path_list = x_path_list[indices]
            y_list = y_list[indices]
        print("gen data path list...", str(len(y_list)))
        return x_path_list, y_list

    def selectOpenImages(self, sample_dict, num, balance, index):
        epoch = self.total_epochs_seen
        sample1, sample2 = self.splitOpenImage(sample_dict)
        if epoch % 13 == 0:
            tmp_index = list(range(len(self.splitOpenImage(sample_dict)[0])))
            np.random.shuffle(tmp_index)
            self.openImageIndex = tmp_index
        sample1_selected = [
            sample1[i]
            for i in self.openImageIndex[(epoch % 13) * 250000:((
                epoch + 1) % 13) * 250000]
        ]
        if balance:
            print(len(sample1), len(sample1_selected), len(sample2),
                  self.num_per_class)
            return np.random.choice(
                sample1_selected + sample2, self.num_per_class, replace=False)
        else:
            return sample1_selected + sample2

    def splitOpenImage(self, sample_dict):
        sample1, sample2 = [], []
        for img in sample_dict:
            if 'openImage' in img:
                sample1.append(img)
            else:
                sample2.append(img)
        return sample1, sample2

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(
            (len(index_array), ) + self.image_shape, dtype=K.floatx())
        #recorded broken files
        broken_files = open(self.broken_recorder, 'r').read().splitlines()
        brokenlist = {}
        for f in broken_files:
            brokenlist[f] = 1
        grayscale = False
        for i, j in enumerate(index_array):
            fname = self.x_path_list[j]
            try:
                img = load_img(
                    fname, grayscale=grayscale, target_size=self.target_size)
                x = img_to_array(img)
                #only random transform during phase TRAIN
                if self.phase == 'train':
                    x = self.image_data_generator.random_transform(x)
                # x = self.image_data_generator.standardize(x)
                x[..., 0] -= 103.939
                x[..., 1] -= 116.779
                x[..., 2] -= 123.68
            except Exception, e:
                #record broken image and random generate x with the same shape of input
                # print("img broken: {}".format(fname))
                if fname not in brokenlist:
                    brokenlist[fname] = 1
                    with open(self.broken_recorder, 'a+') as bf:
                        bf.write(fname + '\n')
                x = np.random.normal(size=self.image_shape)
            batch_x[i] = x

        batch_y = np.zeros((len(batch_x), self.y_dimension), dtype=K.floatx())
        for i, label in enumerate(self.y_list[index_array]):
            batch_y[i, label] = 1.
        if self.nb_output == 1:
            return batch_x, batch_y
        else:
            return batch_x, [batch_y] * self.nb_output

    def next(self):
        with self.lock:
            # index_array, current_index, current_batch_size = next(self.index_generator)
            index_array = next(self.index_generator)
        # if current_index == 0:
        # self.total_epochs_seen += 1
        # if self.is_ergodic_files and self.phase=='train':
        # self.x_path_list, self.y_list = self.gen_ergodic_data_path_list()
        # else:
        # self.x_path_list, self.y_list = self.gen_data_path_list()
        # if len(self.x_path_list) == 0:
        # if self.is_ergodic_files and self.phase=='train':
        # self.x_path_list, self.y_list = self.gen_ergodic_data_path_list()
        # else:
        # self.x_path_list, self.y_list = self.gen_data_path_list()
        return self._get_batches_of_transformed_samples(index_array)


if __name__ == '__main__':
    image_data_generator = MyImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        # rotation_range=8.,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # shear_range=0.3,
        # zoom_range=0.08,
        # horizontal_flip=True,
        rescale=1. / 255)
    batch_size = 16
    is_ergodic_files = None
    train_label_file = '/home/xiziwang/tools/freezed/labeled_freezed_train.txt'
    val_label_file = '/home/xiziwang/tools/freezed/labeled_freezed_val.txt'
    train_gen = image_data_generator.flow_from_label_file(
        train_label_file,
        batch_size=batch_size,
        is_ergodic_files=is_ergodic_files)
    import time
    while 1:
        t1 = time.time()
        next(train_gen)
        print('generate one batch cost {}s'.format(time.time() - t1))
