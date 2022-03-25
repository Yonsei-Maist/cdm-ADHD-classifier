
import re

from network.core import CDMBert

new_origin = ''

total_label_dic = {}
with open('./data/origin.txt', 'r', encoding='utf8') as fp:
    lines = fp.readlines()
    for line in lines:
        split = line.strip().split('\t')

        if len(split) != 2:
            continue

        content = split[0].strip()
        label = split[1].strip()

        label_split = re.split('[/]', label)
        label_dic = {}

        for label_one in label_split:
            label_one_ = label_one.strip()

            split_ = label_one_.split('|')

            if len(label_one_) > 0 and len(split_) == 3:
                if label_one_ not in label_dic:
                    label_dic[label_one_] = ''

        if len(label_dic.keys()) == 1:
            for key in label_dic.keys():
                if key in total_label_dic:
                    total_label_dic[key].append(content)
                else:
                    total_label_dic[key] = [content]

for key in total_label_dic.keys():
    if len(total_label_dic[key]) > 50:
        for content in total_label_dic[key]:
            new_origin += "{0}\t{1}\n".format(content, key)
        print('key: {0}, count: {1}'.format(key, len(total_label_dic[key])))

with open('./data/new_origin.txt', 'w', encoding='utf8') as fp:
    fp.write(new_origin.strip())

data_path, num_classes = CDMBert.create_data_file('./data/new_origin.txt')
