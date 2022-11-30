import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset

from util.data_util import sa_create
from util.data_util import data_prepare


class CIRCUITNET(Dataset):
    def __init__(self, split='train', data_root='trainval', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop


        if split == 'train':
            data_root = data_root + 'trainval'
            data_list = sorted(os.listdir(data_root))
            self.data_list = [item for item in data_list]
        # else:
        #     self.data_list = [item for item in data_list if 'Area_{}'.format(test_area) in item]
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item)
                data = np.load(data_path)  # center_xywh, N*4
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

        ## label
        label_root = '/horizon-bucket/BasicAlgorithm/Users/jialv.zou/CircuitNet/train_congesion/congestion/label/'
        for item in self.data_list:
            item_temp = 'label_' + item
            if not os.path.exists("/dev/shm/{}".format(item_temp)):
                label_path = os.path.join(label_root, item)
                label = np.load(label_path)  # label
                sa_create("shm://{}".format(item_temp), label)
        print("Totally {} label in {} set.".format(len(self.data_idx), split))


    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        label_name = []
        for i in range(len(data_idx)):
            label_name.append(('label_' + self.data_list[i]))

        label_data = SA.attach("shm://{}".format(label_name[data_idx])).copy()
        center_x, center_y, width, height = data[:,0], data[:,1], data[:,2], data[:,3]
        label = label_data
        # coord, feat, label = data_prepare(coord, feat, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        return center_x, center_y, width, height, label

    def __len__(self):
        return len(self.data_idx) * self.loop
