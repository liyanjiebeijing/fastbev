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



def project_corners(box, intrinsic):
    """
    将 3D box 的角点投影到图像平面
    :param box: nuscenes.utils.data_classes.Box
    :param intrinsic: (3x3) 相机内参矩阵
    :return: 投影后的 2D 点 (2x8)
    """
    # 获取 3D 角点 (3x8)
    corners_3d = box.corners()

    # 投影到 2D (齐次坐标)
    points_2d = intrinsic @ corners_3d

    # 归一化，得到像素坐标
    points_2d = points_2d[:2, :] / points_2d[2, :]

    return points_2d


def draw_box(ax, corners_2d, color='g'):
    """
    在 matplotlib 坐标轴上绘制 3D 框的投影
    :param ax: matplotlib axis
    :param corners_2d: (2x8) 投影后的角点
    """
    # NuScenes 的 3D box 角点顺序：
    # (0,1,2,3) 底面四个点，(4,5,6,7) 顶面四个点
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],   # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],   # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]    # 竖线
    ]

    for i1, i2 in connections:
        ax.plot([corners_2d[0, i1], corners_2d[0, i2]],
                [corners_2d[1, i1], corners_2d[1, i2]],
                color=color, linewidth=2)


def vis_by_self():
    # --- 1. 初始化和加载数据 ---
    my_sample = nusc.sample[20] # 换一个样本以便观察
    cam_front_token = my_sample['data']['CAM_FRONT']
    cam_data = nusc.get('sample_data', cam_front_token)
    cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', cam_data['ego_pose_token'])
    img_filepath = nusc.get_sample_data_path(cam_front_token)

    # --- 2. 加载图像 ---
    im = Image.open(img_filepath)
    width, height = im.size

    # --- 3. 核心：遍历标注并进行坐标转换与投影 ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(im)

    # 遍历该样本中的所有标注 token
    for ann_token in my_sample['anns']:
        ann_record = nusc.get('sample_annotation', ann_token)

        # 使用 devkit 的 Box 类来方便地处理标注框
        box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']))

        # a) 将 Box 从世界坐标系转换到自车坐标系
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        # b) 将 Box 从自车坐标系转换到相机坐标系
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        # c) 投影到图像平面并绘制
        pts_2d = project_corners(box, np.array(cs_record['camera_intrinsic']))
        if np.all((pts_2d[0, :] >= 0) & (pts_2d[0, :] < width) &
                  (pts_2d[1, :] >= 0) & (pts_2d[1, :] < height)):
            box.render(ax, view=np.array(cs_record['camera_intrinsic']), normalize=False)
            # draw_box(ax, pts_2d, color='g') #自己绘制


    ax.set_title(f"CAM_FRONT manual rendering for sample_token: {my_sample['token']}")
    ax.axis('off')
    # plt.show()
    fig.savefig("./cam_front_manual_render.png") # 如果需要保存


if __name__ == '__main__':
    # vis_by_api()
    vis_by_self()