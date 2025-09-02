import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from PIL import Image


def get_2d_corners(box, intrinsic):
    """将 3D box 投影到图像平面"""
    corners_3d = box.corners()  # (3,8)
    pts_2d = intrinsic @ corners_3d
    pts_2d = pts_2d[:2, :] / pts_2d[2, :]
    return pts_2d


def draw_box(ax, corners_2d, img_size, color='g'):
    """
    绘制 3D box 的投影 (只绘制图像范围内的线段)
    :param ax: matplotlib axis
    :param corners_2d: (2x8) 投影后的角点
    :param img_size: (width, height)
    :param color: 颜色
    """
    width, height = img_size
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],   # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],   # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]    # 竖线
    ]

    def in_img(x, y):
        return 0 <= x < width and 0 <= y < height

    for i1, i2 in connections:
        x1, y1 = corners_2d[0, i1], corners_2d[1, i1]
        x2, y2 = corners_2d[0, i2], corners_2d[1, i2]

        # 只有两个端点都在图像范围内，才绘制
        if in_img(x1, y1) and in_img(x2, y2):
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=2)


def visualize_sample(nusc, sample_token, cam_name="CAM_FRONT"):
    sample = nusc.get("sample", sample_token)
    cam_token = sample["data"][cam_name]
    cam_data = nusc.get("sample_data", cam_token)
    cs_record = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", cam_data["ego_pose_token"])
    img_path = nusc.get_sample_data_path(cam_token)

    # 读取图像
    im = Image.open(img_path)
    width, height = im.size
    print(im.size)

    # matplotlib figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(im)

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        box = Box(ann["translation"], ann["size"], Quaternion(ann["rotation"]))

        # global -> ego
        box.translate(-np.array(pose_record["translation"]))
        box.rotate(Quaternion(pose_record["rotation"]).inverse)

        # ego -> camera
        box.translate(-np.array(cs_record["translation"]))
        box.rotate(Quaternion(cs_record["rotation"]).inverse)

        if box.center[2] > 0:
            pts_2d = get_2d_corners(box, np.array(cs_record["camera_intrinsic"]))

            # 过滤：至少有一个点在图像范围内才绘制
            if np.all((pts_2d[0, :] >= 0) & (pts_2d[0, :] < width) &
                      (pts_2d[1, :] >= 0) & (pts_2d[1, :] < height)):
                draw_box(ax, pts_2d, (width, height), color='g')

    ax.set_title(f"{cam_name} with 3D boxes")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig("./cam_front_manual_render.png") # 如果需要保存
    print("图像已成功渲染并保存至 ./cam_front_manual_render.png")


if __name__ == "__main__":
    # 修改为你自己的 NuScenes 数据路径
    dataroot = "/mnt/share_disk/liyanjie/project/Fast-BEV/data/nuscenes_v1.0_mini"
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=True)

    # 取一个样本可视化
    sample_token = nusc.sample[20]["token"]
    visualize_sample(nusc, sample_token, cam_name="CAM_FRONT")
