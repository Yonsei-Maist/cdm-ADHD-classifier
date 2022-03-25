import tensorflow as tf
from enum import Enum
import math
import os
import random
from abc import *


class LOSS(Enum):
    MSE = 1
    COSINE_SIMILARITY = 2
    CATEGORICAL_CROSSENTROPY = 3
    BINARY_CROSSENTROPY = 4
    SPARSE_CATEGORICAL_CROSSENTROPY = 5
    MAE = 6


class METRICS(Enum):
    ACCURACY = 1,
    CATEGORICAL_ACCURACY = 2,
    SPARSE_CATEGORICAL_ACCURACY = 3,
    MEAN = 4


class AvgLogger:
    def __init__(self, avg_list: list):
        self.__avg_type_list = avg_list
        self.__avg_list = None
        self.__build_avg()

    def __build_avg(self):
        self.__avg_list = []

        for avg_type in self.__avg_type_list:
            self.__avg_list.append(tf.keras.metrics.Mean(avg_type, dtype=tf.float32))

    def refresh(self):
        self.__build_avg()

    def update_state(self, value_list: list):
        if len(value_list) != len(self.__avg_type_list):
            raise Exception(
                "Unmatch Length: {0} compared to type list length({1})".format(len(value_list),
                                                                               len(self.__avg_type_list)))

        for i in range(len(self.__avg_list)):
            avg = self.__avg_list[i]
            value = value_list[i]
            avg.update_state(value)

    def result(self):
        return ",".join("{0}: {1}".format(self.__avg_type_list[i], self.__avg_list[i].result().numpy())
                        for i in range(len(self.__avg_type_list)))

    def result_value(self):
        return [self.__avg_list[i].result().numpy() for i in range(len(self.__avg_type_list))]


class Dataset:
    def __init__(self, batch_size, model=None, input_dtype=tf.float32, output_dtype=tf.float32):
        self.__inputs = None
        self.__labels = None
        self.__origins = None
        self.batch_size = batch_size
        self.__model = model
        self.input_dtype = input_dtype
        self.output_dtype = output_dtype

    def set(self, inputs, labels, origin_file=None):
        len_input = -1
        for i in range(len(inputs)):
            if i == 0:
                len_input = len(inputs[i])
            elif len_input != len(inputs[i]):
                raise Exception("Doesn't match a length of all inputs")

        for i in range(len(labels)):
            if len_input != len(labels[i]):
                raise Exception("Doesn't match between the length of inputs and a length of labels")

        self.__inputs = inputs
        self.__labels = labels
        self.__origins = origin_file

    def output_transform(self, item):
        return tf.convert_to_tensor(item, dtype=self.output_dtype)

    def input_transform(self, model, inputs):

        if model is None:
            return [tf.convert_to_tensor(item, dtype=self.input_dtype) for item in inputs]
        else:
            len_batch = len(inputs[0])
            batch_result = []

            for i in range(len_batch):
                res = model.data_transform([item[i] for item in inputs])
                for j in range(len(res)):
                    if len(batch_result) <= j:
                        batch_result.append([])

                    batch_result[j].append(res[j])

            return [tf.convert_to_tensor(item, dtype=self.input_dtype) for item in batch_result]

    def get_all(self):
        """
        use in tensorflow
        :return: all data
        """
        if len(self.__labels) == 1:
            return self.input_transform(self.__model, self.__inputs), self.output_transform(self.__labels[0])
        else:
            return self.input_transform(self.__model, self.__inputs), [self.output_transform(item) for item in self.__labels]

    def get(self):
        """
        use in pytorch
        :return: iteration data
        """
        iter = math.ceil(len(self) / self.batch_size)

        for i in range(iter):
            raw_input = [item[i * self.batch_size: i * self.batch_size + self.batch_size] for item in
                       self.__inputs]
            if len(self.__labels) == 1:
                yield self.input_transform(self.__model, raw_input), \
                      self.output_transform(self.__labels[0][i * self.batch_size: i * self.batch_size + self.batch_size])
            else:
                yield self.input_transform(self.__model, raw_input), \
                      [self.output_transform(item[i * self.batch_size: i * self.batch_size + self.batch_size]) for item in self.__labels]

    def get_origin(self):
        if self.__origins is None:
            raise Exception("the origin file is empty")

        iter = math.ceil(len(self) / self.batch_size)
        for i in range(iter):
            yield self.__origins[i * self.batch_size: i * self.batch_size + self.batch_size]

    def __len__(self):
        if self.__inputs is None:
            return 0

        if len(self.__inputs) > 0:
            return len(self.__inputs[0])

        return 0


class DatasetFactory:
    @staticmethod
    def get_dtype(value):
        if isinstance(value, int):
            return tf.int32
        else:
            return tf.float32

    @staticmethod
    def make_dataset(train_data, test_data, data_all, sp_ratio, is_classify):

        item_one = data_all[0]

        data_train = []
        data_validation = []
        data_test = []

        if is_classify:
            dic_label = {}

            for data in data_all:
                label_one = data['output']
                if label_one in dic_label:
                    dic_label[label_one].append(data)
                else:
                    dic_label[label_one] = [data]

            for key in dic_label.keys():
                data_arr = dic_label[key]
                sp = int(len(data_arr) * sp_ratio)
                random.shuffle(data_arr)

                data_train.extend(data_arr[:sp])
                data_test.extend(data_arr[sp:])
        else:
            sp = int(len(data_all) * sp_ratio)
            data_train = data_all[:sp]
            data_test = data_all[sp:]

        random.shuffle(data_train)
        random.shuffle(data_test)

        input_train = None
        input_test = None
        output_train = None
        output_test = None
        origin_train = None
        origin_test = None

        if isinstance(item_one['input'], dict):
            input_train = [[item['input'][i] for item in data_train] for i in range(len(item_one['input'].keys()))]
            input_test = [[item['input'][i] for item in data_test] for i in range(len(item_one['input'].keys()))]
        else:
            input_train = [[item['input'] for item in data_train]]
            input_test = [[item['input'] for item in data_test]]

        if isinstance(item_one['output'], dict):
            output_train = [[item['output'][i] for item in data_train] for i in range(len(item_one['output'].keys()))]
            output_test = [[item['output'][i] for item in data_test] for i in range(len(item_one['output'].keys()))]
        else:
            output_train = [[item['output'] for item in data_train]]
            output_test = [[item['output'] for item in data_test]]

        if 'origin' in item_one:
            origin_train = [item['origin'] for item in data_train]
            origin_test = [item['origin'] for item in data_test]

        train_data.set(input_train, output_train, origin_train)
        test_data.set(input_test, output_test, origin_test)

        return train_data, test_data


class ModelCore(metaclass=ABCMeta):
    def __init__(self, data_path, save_path, batch_size=64, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=None,
                 optimizer=None, lr=0.001,
                 train_test_ratio=0.8, validation_ratio=0.1, is_classify=False, loss_weights=None,
                 input_dtype=tf.float32, output_dtype=tf.float32):

        if metrics is None:
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self._train_data = Dataset(batch_size, self, input_dtype, output_dtype)
        self._test_data = Dataset(batch_size, self, input_dtype, output_dtype)
        self._validation_data = Dataset(batch_size, self, input_dtype, output_dtype)

        self.input_dtype = input_dtype
        self.output_dtype = output_dtype
        self.model = None
        self.batch_size = batch_size
        self._data_path = data_path
        self._save_path = save_path
        self._data_all = None
        self._train_test_ratio = train_test_ratio
        self._validation_ratio = validation_ratio
        self._is_classify = is_classify
        self.loss_weights = loss_weights

        self.is_multi_output = isinstance(loss, list)
        self.loss = loss
        self.metrics = metrics
        self.optimizer = optimizer

        self.read_data()
        self.make_dataset()
        self.build_model()
        self.compile_model()

    def check_integer_string(self, int_value):
        try:
            return int(int_value)
        except ValueError as e:
            return -1

    def get_train_data(self):
        return self._train_data

    def get_test_data(self):
        return self._test_data

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def read_data(self):
        pass

    def make_dataset(self):
        if self._data_all is not None and len(self._data_all) > 0:
            self._train_data, self._test_data = \
                DatasetFactory.make_dataset(self._train_data, self._test_data,
                                            self._data_all,
                                            self._train_test_ratio,
                                            self._is_classify)

    def compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics, loss_weights=self.loss_weights)

    def data_transform(self, item):
        return item

    def load_weight(self):
        """
        load weight without checkpoint
        """
        pass

    def train(self, epoch=1000, save_each_epoch=100, callbacks=None):
        x_train, y_train = self._train_data.get_all()

        must_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(self._save_path, 'ckpt_{epoch:04d}.tf'),
            save_freq='epoch',
            monitor='loss',
            save_weights_only=True)

        callback_list = [must_callback]

        if callbacks is not None:
            callback_list.extend(callbacks)

        self.model.fit(x_train, y_train, batch_size=self.batch_size, epochs=epoch,
                       validation_split=self._validation_ratio, callbacks=callback_list)

    def test(self, epoch=None):
        x_test, y_test = self._test_data.get_all()

        if epoch is not None:
            self.model.load_weights(os.path.join(self._save_path, "ckpt_{epoch:04d}.tf".format(epoch=epoch)))

            self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
        else:
            file_list = os.listdir(self._save_path)
            for file in file_list:
                idx = file.find('.tf')
                if file.find('.tf') > -1:
                    sub_file = file[:idx + 4]
                    self.model.load_weights(os.path.join(self._save_path, sub_file))

                    self.model.evaluate(x_test, y_test, batch_size=self.batch_size)
