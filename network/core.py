from network.lib import ModelCore
import tensorflow as tf
import random
import tensorflow_hub as hub
from official.nlp import optimization

import os
import re


class CDMBert(ModelCore):

    @staticmethod
    def create_data_file(origin_path):
        labels = {}
        label_contents = ''
        datas = ''

        parent = os.path.dirname(origin_path)

        with open(origin_path, 'r', encoding='utf8') as fp:
            lines = fp.readlines()
            for line in lines:
                split = line.strip().split('\t')

                if len(split) != 2:
                    continue

                data = split[0]
                label = split[1]

                if label == "-" or label == "" or label == "p" or label == "--":
                    continue

                if label in labels:
                    label_data = labels[label]
                    index = label_data["index"]
                else:
                    index = len(labels.keys())
                    labels[label] = {'index': index}
                    label_contents = "{0}{1}\t{2}\n".format(label_contents, index, label)

                datas = "{0}{1}\t{2}\n".format(datas, data, index)

        data_path = os.path.join(parent, 'train.txt')

        with open(os.path.join(parent, 'train.txt'), 'w', encoding='utf8') as data_fp:
            data_fp.write(datas)

        with open(os.path.join(parent, 'label.txt'), 'w', encoding='utf8') as label_fp:
            label_fp.write(label_contents)

        return data_path, len(labels.keys())

    @staticmethod
    def clean_text(text):
        return re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]', " ", text)


class TFHubBert(ModelCore):
    def __init__(self, data_path, num_classes):
        self.tfhub_handle_encoder = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3"
        self.tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3"
        self.num_classes = num_classes

        steps_per_epoch = 867
        num_train_steps = steps_per_epoch * 10
        num_warmup_steps = int(0.1 * num_train_steps)

        optimizer = optimization.create_optimizer(init_lr=3e-5,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        ModelCore.__init__(self, save_path="./bert", data_path=data_path, train_test_ratio=0.9,
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=[tf.keras.metrics.BinaryAccuracy()],
                           optimizer=optimizer,
                           is_classify=True, input_dtype=tf.string, batch_size=8, lr=0.001, output_dtype=tf.int32)

    def build_model(self):
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
        preprocessing_layer = hub.KerasLayer(self.tfhub_handle_preprocess, name='preprocessing')
        encoder_inputs = preprocessing_layer(text_input)
        encoder = hub.KerasLayer(self.tfhub_handle_encoder, trainable=True, name='BERT_encoder')
        outputs = encoder(encoder_inputs)
        net = outputs['pooled_output']

        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(self.num_classes, activation=None, name='classifier')(net)
        self.model = tf.keras.Model(text_input, net)

    def read_data(self):
        self._data_all = []
        with open(self._data_path, 'r', encoding='utf8') as fp:
            lines = fp.readlines()
            for line in lines:
                content_split = line.strip().split('\t')

                if len(content_split) != 2:
                    continue

                input_text = content_split[0]
                label = int(content_split[1])

                zero = label

                self._data_all.append({'input': CDMBert.clean_text(input_text), 'output': zero})

        random.shuffle(self._data_all)
