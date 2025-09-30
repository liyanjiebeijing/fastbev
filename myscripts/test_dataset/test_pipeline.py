import argparse
import random
import mmcv
import numpy as np
from mmcv.parallel import DataContainer
import torch
import matplotlib.pyplot as plt
import copy
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import cv2

# 注册项目特定的模块 (如果您的数据集中有自定义类)
# from projects.fast_bev.datasets import NuScenesMultiView_Map_Dataset2
# from projects.fast_bev.datasets.pipelines import RandomAugImageMultiViewImage, KittiSetOrigin, NormalizeMultiviewImage, MultiViewPipeline


def unnormalize_image(img, mean, std):
    """反归一化图像以便显示"""
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * std) + mean
    # 确保图像数据类型正确以便显示
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def project_points(points_3d, projection_matrix):
    """将 3D 点投影到 2D 图像平面"""
    # 将 3D 点转换为齐次坐标 (num_points, 4)
    points_4d = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

    # 使用投影矩阵进行变换
    points_2d_proj = points_4d @ projection_matrix.T

    # 过滤掉在相机后方的点 (z < epsilon)
    mask = points_2d_proj[:, 2] > 1e-3
    points_2d_proj = points_2d_proj[mask]

    if points_2d_proj.shape[0] == 0:
        return None, None

    # 齐次坐标归一化，得到像素坐标
    points_2d = points_2d_proj[:, :2] / points_2d_proj[:, 2:3]
    return points_2d, mask

def draw_3d_boxes_on_image(image, boxes_3d, projection_matrix, color=(0, 255, 0)):
    """在单张图像上绘制投影后的所有3D边界框"""
    if boxes_3d.tensor.numel() == 0:
        return image

    # 获取所有框的8个角点 (num_boxes, 8, 3)
    corners_3d = boxes_3d.corners.cpu().numpy()

    # 3D 框的12条边连接关系
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # 底面
        (4, 5), (5, 6), (6, 7), (7, 4),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)   # 侧边
    ]

    # 确保图像是可写的
    image_draw = image.copy()

    for box_corners in corners_3d:
        # 将当前框的8个角点投影到2D图像
        corners_2d, mask = project_points(box_corners, projection_matrix)

        if corners_2d is None:
            continue

        # 绘制12条边
        for i, j in edges:
            # 只有当边的两个端点都有效（在相机前方）时才绘制
            if mask[i] and mask[j]:
                # 根据mask的索引找到在corners_2d中的对应点
                p1_idx = np.sum(mask[:i+1]) - 1
                p2_idx = np.sum(mask[:j+1]) - 1

                # 获取2D坐标点并转换为整数
                p1 = tuple(corners_2d[p1_idx].astype(int))
                p2 = tuple(corners_2d[p2_idx].astype(int))

                # 使用OpenCV绘制线条
                cv2.line(image_draw, p1, p2, color, 1)

    return image_draw



def to_tensor(data):
    """递归解包 DataContainer / Tensor / list / dict 为 numpy."""
    if isinstance(data, DataContainer):
        return to_tensor(data.data)   # DataContainer 取出实际内容
    elif torch.is_tensor(data):
        return data.cpu()
    elif isinstance(data, list):
        return [to_tensor(d) for d in data]
    elif isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    else:
        return data


def main():
    # ----------------------------
    # 1. 解析参数并加载配置
    # ----------------------------
    parser = argparse.ArgumentParser(description='Visualize a single sample from a 3D dataset pipeline')
    parser.add_argument('config', help='path to the config file')
    parser.add_argument('--index', type=int, default=None, help='Index of the sample to visualize. Random if not provided.')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)

    # MMDetection3D v0.17.x 风格的配置
    # 如果您的配置是 MMEngine 风格 (MMDet v3.x), 可能需要调整为 cfg.train_dataloader.dataset
    dataset_cfg = cfg.data.train.dataset
    # dataset_cfg = cfg.data.val

    # ----------------------------
    # 2. 构建数据集
    # ----------------------------
    # 注意: 为了可视化，我们需要原始图像和3D标注，
    # 因此需要从 pipeline 中移除格式化和收集的步骤。
    # 我们将创建一个新的pipeline用于可视化。
    # vis_pipeline_cfg = [item for item in dataset_cfg.pipeline if item['type'] not in ['DefaultFormatBundle3D', 'Collect3D']]
    # dataset_cfg.pipeline = vis_pipeline_cfg

    collect_3d_pipeline = None
    for each in dataset_cfg.pipeline:
        if each['type'] == 'Collect3D':
            collect_3d_pipeline = each
            break

    required_keys = ['lidar2img', 'filename']
    for reuqired_key in required_keys:
        if collect_3d_pipeline is not None and reuqired_key not in collect_3d_pipeline['keys']:
            collect_3d_pipeline['keys'].append(reuqired_key)

     # 构建数据集
    dataset = build_dataset(dataset_cfg)
    print("Dataset type:", type(dataset))
    print("Dataset length:", len(dataset))
    print("Class names:", dataset.CLASSES)

    # ----------------------------
    # 3. 获取一个经过 pipeline 处理的样本
    # ----------------------------
    if args.index is not None:
        idx = args.index
    else:
        idx = random.randint(0, len(dataset) - 1)



    raw_sample = dataset[idx]

    # import pdb
    # pdb.set_trace()

    sample = to_tensor(raw_sample)   # 解包成 numpy-friendly 格式
    print(f"\nVisualizing sample index: {idx}")


    # 从样本中提取数据
    imgs = sample['img'] # 此时是原始图像列表
    # img_metas = sample['img_metas']
    gt_bboxes_3d = sample['gt_bboxes_3d'] # LiDARInstance3DBoxes 对象
    # import pdb
    # pdb.set_trace()

    # ----------------------------
    # 4. 可视化
    # ----------------------------
    num_views = len(imgs)
    # 获取归一化参数，如果存在的话
    norm_cfg = next((t for t in cfg.data.val.pipeline if t['type'] == 'NormalizeMultiviewImage'), None)
    mean = np.array(norm_cfg['mean'], dtype=np.float32)
    std = np.array(norm_cfg['std'], dtype=np.float32)

    drawn_imgs = []
    for i in range(num_views):
        # 图像已经通过 pipeline 处理（例如，可能已经缩放和填充）
        # 如果您的 pipeline 中有 NormalizeMultiviewImage，则需要反归一化
        # 这里假设 NormalizeMultiviewImage 返回的是 Tensor，并且 to_rgb=True
        img_tensor = copy.deepcopy(imgs[i])
        img_unnormalized = unnormalize_image(img_tensor, mean, std)

        # 获取 lidar 到第 i 个相机的投影矩阵
        lidar2img_matrix = sample['lidar2img']['extrinsic'][i]

        # 在图像上绘制所有3D框
        img_with_boxes = draw_3d_boxes_on_image(img_unnormalized, gt_bboxes_3d, lidar2img_matrix)
        drawn_imgs.append(img_with_boxes)

    # 显示结果
    n = len(drawn_imgs)
    cols = min(n, 6)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(36, 3 * rows))
    for i, img in enumerate(drawn_imgs):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img) # mmcv.bgr2rgb(img) for opencv default
        plt.title(sample['img_metas']['img_info'][i]['filename'].split('/')[-1].split('__15')[1])
        # plt.title(sample['img_metas']['img_info'][i]['filename'].split('/')[-1])
        plt.axis("off")
    plt.suptitle(f"Sample {idx} - 3D BBoxes Projection", fontsize=16)
    output_path = f"./sample_{idx}_visualization.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


if __name__ == "__main__":
    main()