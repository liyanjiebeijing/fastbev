from nuscenes.nuscenes import NuScenes

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


nuscence_root_dir="/mnt/share_disk/liyanjie/project/Fast-BEV/data/nuscenes_v1.0_mini"
nusc = NuScenes(version='v1.0-mini', dataroot=nuscence_root_dir, verbose=True)


def vis_by_api():
    # 选取一个你想要可视化的样本 (这里用第一个样本作为例子)
    my_sample = nusc.sample[20]

    # 获取前置摄像头的数据 token
    cam_front_token = my_sample['data']['CAM_FRONT']

    # 使用 devkit 的渲染函数
    # 它会自动处理所有坐标转换、投影和绘制工作
    # with_anns=True 表示绘制标注框
    nusc.render_sample_data(cam_front_token, with_anns=True, out_path='./cam_front_with_anns.png')

    print("图像已成功渲染并保存至 ./cam_front_with_anns.png")



if __name__ == '__main__':
    vis_by_api()