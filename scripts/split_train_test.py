import numpy as np
import os
import shutil
from tqdm import *

def split_train_test(data, test_ratio):
    # 设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(33)
    # permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))
    # test_ratio为测试集所占的半分比
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return train_indices, test_indices


# 测试
root_dir = '/home/users/jialv.zou/datasets/congestion_point/point_cloud/xywh/'
train_dir = '/home/users/jialv.zou/datasets/congestion_point/point_cloud/trainval/'
test_dir = '/home/users/jialv.zou/datasets/congestion_point/point_cloud/test/'
filename = os.listdir(root_dir)
train_index, test_index = split_train_test(filename, 0.1)

for i in tqdm(range(len(train_index))):
    source = root_dir + filename[train_index[i]]
    target = train_dir + filename[train_index[i]]
    shutil.copyfile(source, target)

for j in tqdm(range(len(test_index))):
    source = root_dir + filename[test_index[j]]
    target = test_dir + filename[test_index[j]]
    shutil.copyfile(source, target)



# print(len(train_set), "train +", len(test_set), "test")