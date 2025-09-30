import mmcv
import torch
import cv2
import numpy as np
import os
from mmcv import Config
from mmdet3d.apis import init_model, inference_detector, show_result_meshlab
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.apis import single_gpu_test
from mmcv.parallel import DataContainer
from mmdet3d.core.visualizer import show_multi_modality_result
import copy
import matplotlib.pyplot as plt



def print_dict_tree(data, indent=0, prefix='', processed_keys=None):
    """
    递归打印字典的树形结构，支持 DataContainer，处理键名前4字符相同的情况
    :param data: 要处理的数据(字典或列表)
    :param indent: 当前缩进级别
    :param prefix: 连接线前缀
    :param processed_keys: 已处理的键名前缀集合
    """
    if processed_keys is None:
        processed_keys = set()

    # ✅ 新增对 DataContainer 的特殊处理
    if isinstance(data, DataContainer):
        # 打印 DataContainer 的数据类型和形状
        inner_data = data.data
        if isinstance(inner_data, np.ndarray):
            print(prefix + f'└── DataContainer (ndarray) shape={inner_data.shape}')
        elif isinstance(inner_data, torch.Tensor):
            print(prefix + f'└── DataContainer (Tensor) shape={inner_data.shape}')
        elif isinstance(inner_data, (list, dict)):
            print(prefix + f'└── DataContainer (type={type(inner_data).__name__})')
            # 递归打印内部数据
            print_dict_tree(inner_data, indent, prefix + '│   ', processed_keys)
        else:
            len_str = f', len={len(inner_data)}' if hasattr(inner_data, '__len__') else ''
            print(prefix + f'└── DataContainer (type={type(inner_data).__name__}, len={len_str})')
        return

    if isinstance(data, dict):
        keys = list(data.keys())
        for i, key in enumerate(keys):
            is_last = (i == len(keys) - 1)
            connector = '└── ' if is_last else '├── '

            key_prefix = str(key)[:4]
            skip_recursion = key_prefix in processed_keys
            if not skip_recursion:
                processed_keys.add(key_prefix)

            print(prefix + connector + str(key))

            if not skip_recursion:
                new_prefix = prefix + ('    ' if is_last else '│   ')
                print_dict_tree(data[key], indent + 1, new_prefix, processed_keys)

    elif isinstance(data, list) and len(data) > 0:
        id = len(data) // 2
        print(prefix + f'├── [{id}]')
        print_dict_tree(data[id], indent + 1, prefix + '│   ', processed_keys)

    else:
        if isinstance(data, np.ndarray):
            print(prefix + f'└── ndarray shape={data.shape}')
        else:
            try:
                length = len(data)
            except Exception:
                length = None
            if length is not None:
                print(prefix + f'└── type={type(data).__name__}, len={length}')
            else:
                print(prefix + f'└── type={type(data).__name__}, value={data}')



def to_tensor(batch):
    if isinstance(batch, DataContainer):
        return batch.data[0]   # 取出实际的 Tensor
    elif isinstance(batch, list):
        return [to_tensor(b) for b in batch]
    elif isinstance(batch, dict):
        return {k: to_tensor(v) for k, v in batch.items()}
    else:
        return batch


def unnormalize_image(img, mean, std):
    """反归一化图像以便显示"""
    # img = img.permute(1, 2, 0).cpu().numpy()
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



# ============ Step 1: 加载配置和模型 ============
cfg_file = 'configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py'  # 你的配置文件
ckpt_file = 'work_dirs/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4/epoch_20.pth'  # 训练好的模型路径

cfg = Config.fromfile(cfg_file)
# cfg.data.test.test_mode = True   # 设置为测试模式

# 初始化模型
model = init_model(cfg, ckpt_file, device='cuda:0')

# ============ Step 2: 构建验证集 DataLoader ============
dataset = build_dataset(cfg.data.val)
data_loader = build_dataloader(
    dataset,
    samples_per_gpu=1,
    workers_per_gpu=2,
    dist=False,
    shuffle=False
)

# ============ Step 3: 推理 ============
out_dir = './vis_results'
os.makedirs(out_dir, exist_ok=True)

model.eval()
for i, data in enumerate(data_loader):
    data = to_tensor(data)

    # 放到 GPU 上
    data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)

    print_dict_tree(result)

    # ===== Step 4: 可视化（绘制到图像上） =====
    imgs = data['img'].data[0][:6].permute(0,2,3,1).cpu().numpy()  # (6, H, W, 3)



    # 取出 GT 3D box
    gt_bboxes_3d = data['gt_bboxes_3d'][0] if 'gt_bboxes_3d' in data else None

    # 取出预测 3D box（FastBEV 输出是一个 list，每帧一个 result dict）
    pred_bboxes_3d = result[0]['boxes_3d']
    scores_3d = result[0]['scores_3d']

    score_thr = 0.3
    mask = scores_3d > score_thr
    pred_bboxes_3d = pred_bboxes_3d[mask]

    # 相机投影矩阵
    proj_mat = data['img_metas'][0]['lidar2img']['extrinsic']

    import pdb
    pdb.set_trace()

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
        img_tensor = copy.deepcopy(imgs[i])
        img_unnormalized = unnormalize_image(img_tensor, mean, std)

        # 获取 lidar 到第 i 个相机的投影矩阵
        lidar2img_matrix = proj_mat[i]

        # 在图像上绘制所有3D框
        img_with_boxes = draw_3d_boxes_on_image(img_unnormalized, pred_bboxes_3d, lidar2img_matrix, color=(0, 255, 0))
        if gt_bboxes_3d is not None:
            img_with_boxes = draw_3d_boxes_on_image(img_with_boxes, gt_bboxes_3d, lidar2img_matrix, color=(255, 0, 0))
        drawn_imgs.append(img_with_boxes)


    for i in range(num_views):
        cv2.imwrite(f"./{out_dir}/view_{i}.jpg", drawn_imgs[i][:, :, ::-1])


    # 显示结果
    n = len(drawn_imgs)
    assert n==6
    plt.figure(figsize=(18, 8))
    for i, img in enumerate(drawn_imgs):
        plt.subplot(2, 3, i + 1)
        plt.imshow(img)
        full_name = data['img_metas'][0]['img_info'][i]['filename'].split('/')[-1]
        short_names = full_name.split("__")
        short_name = short_names[-2] + "__" + short_names[-1]
        plt.title(short_name)
        plt.axis("off")

    plt.suptitle(f"3D BBoxes Projection", fontsize=16)
    output_path = f"{out_dir}/sample_visualization.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")


    break



    # # 可视化（会生成多张 6V 图像）
    # show_multi_modality_result(
    #     img=imgs,
    #     gt_bboxes=gt_bboxes_3d,
    #     pred_bboxes=pred_bboxes_3d,
    #     proj_mat=proj_mat,
    #     out_dir=out_dir,
    #     filename=f'frame_{i:06d}',
    #     box_mode='lidar',
    #     img_metas=data['img_metas'][0].data[0],
    #     show=False
    # )


    # if i > 10:  # 只跑前10张看看效果
    #     break

print("推理与可视化完成，结果保存在 vis/ 目录下。")
