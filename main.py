from network.core import CDMBert, TFHubBert


"""
change to real data path
"""
data_path, num_classes = CDMBert.create_data_file('./data/new_origin.txt')

core = TFHubBert(data_path, num_classes=1)

"""
remove # if you want to train
"""
# core.train(epoch=10, save_each_epoch=1)

for i in range(1, 10):
    core.test(i)

