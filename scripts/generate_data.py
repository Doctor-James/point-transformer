import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
import tqdm
from tqdm import trange
import multiprocessing
import time
from multiprocessing import shared_memory



def count_area(filename):
    # print("task", i)
    area_sum = np.empty(shape=[1])
    instance_placement = np.load(filename, allow_pickle=True).item()
    point3ds = np.empty(shape=[1,3])
    for key, value in instance_placement.items():
        yt = value[0]
        xl = value[1]
        yb = value[2]
        xr = value[3]
        center_x = (xr + xl)/2
        center_y = (yb + yt)/2
        area = abs((yt-yb)*(xr-xl))
        area = int(area)
        # if area <50:
        # point3d = np.array([center_x,center_y,area])
        # point3d = np.expand_dims(point3d,axis=0)
        # point3ds = np.append(point3ds,point3d,axis=0)
        area_sum = np.append(area_sum,[area],axis=0)
    # point3ds = point3ds[1:,]
    area_sum = area_sum[1:]
    return area_sum

def trans2npy(filename):
    root_path = '/horizon-bucket/BasicAlgorithm/Users/jialv.zou/CircuitNet/CircuitNet/graph_features_new2/instance_placement/'
    output_dir = '/home/users/jialv.zou/datasets/congestion_point/point_cloud/xywh/'
    output_path = output_dir + filename
    input_path = root_path + filename
    instance_placement = np.load(input_path, allow_pickle=True).item()
    point3ds = np.empty(shape=[1,4])
    for key, value in instance_placement.items():
        yt = value[0]
        xl = value[1]
        yb = value[2]
        xr = value[3]
        width = (xr-xl)
        height = (yb-yt)
        center_x = (xr + xl)/2
        center_y = (yb + yt)/2
        area = abs(width*height)
        if area <80:
            point3d = np.array([center_x,center_y,width,height])
            point3d = np.expand_dims(point3d,axis=0)
            point3ds = np.append(point3ds,point3d,axis=0)

    point3ds = point3ds[1:]
    np.save(output_path,point3ds)



if __name__ == '__main__':
    # root_path = '/horizon-bucket/BasicAlgorithm/Users/jialv.zou/CircuitNet/CircuitNet/graph_features_new2/instance_placement/'
    # filename = os.listdir(root_path)
    #
    # begin_time = time.time()
    # area_ = np.empty(shape=[1])
    #
    # file = []
    # for i in range(len(filename)):
    #     if(i==9579):
    #         continue
    #     file.append(root_path + filename[i])
    #
    # #多进程
    # with multiprocessing.Pool(64) as p:
    #     ret = list(tqdm.tqdm(p.imap(count_area, file), total=len(file)))
    #     p.close()  # 关闭进程池
    #     p.join()  # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭
    #
    # for j in range(len(ret)):
    #     area_ = np.append(area_,ret[j],axis=0)
    #
    # end_time = time.time()
    # print("parallel time: ", end_time - begin_time)
    #
    # area_ = area_[1:]
    # arr_gb = area_.flatten()  # 数组转为1维
    # arr_gb = pd.Series(arr_gb)  # 转换数据类型
    # arr_gb = arr_gb.value_counts()  # 计数
    # # arr_gb.sort_index(inplace=True)  # 排序
    # print(arr_gb)

    #点云转换
    root_path = '/horizon-bucket/BasicAlgorithm/Users/jialv.zou/CircuitNet/CircuitNet/graph_features_new2/instance_placement/'
    output_path = '/home/users/jialv.zou/datasets/congestion_point/point_cloud/xywh/'
    filename = os.listdir(root_path)

    begin_time = time.time()
    area_ = np.empty(shape=[1])

    file = []
    for i in range(len(filename)):
        if(i==9579):
            continue
        file.append(filename[i])

    #多进程
    with multiprocessing.Pool(64) as p:
        ret = list(tqdm.tqdm(p.imap(trans2npy, file), total=len(file)))
        p.close()  # 关闭进程池
        p.join()  # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭






