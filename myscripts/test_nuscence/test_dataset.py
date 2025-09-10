import os
import random
import mmcv
import numpy as np
import matplotlib.pyplot as plt

from mmcv import Config
from mmdet3d.datasets import build_dataset

def show_images(imgs, title="Sample Multiview Images"):
    """简单可视化多相机图像"""
    n = len(imgs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(mmcv.bgr2rgb(img))
        plt.axis("off")
    plt.suptitle(title)
    plt.savefig("./data/test_dataset.png")


def main():
    # ----------------------------
    # 1. 配置（直接拷贝过来的 dict）
    # ----------------------------
    class_names = [
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
    ]


    data_config = {
        'src_size': (900, 1600),
        'input_size': (256, 704),
        # train-aug
        'resize': (-0.06, 0.11),
        'crop': (-0.05, 0.05),
        'rot': (-5.4, 5.4),
        'flip': True,
        # test-aug
        'test_input_size': (256, 704),
        'test_resize': 0.0,
        'test_rotate': 0.0,
        'test_flip': False,
        # top, right, bottom, left
        'pad': (0, 0, 0, 0),
        'pad_divisor': 32,
        'pad_color': (0, 0, 0),
    }

    input_modality = dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=True)

    dataset_cfg = dict(
        type='NuScenesMultiView_Map_Dataset2',  # 这里替换成你的 dataset_type
        data_root='data/nuscenes/',
        pipeline=[
            dict(type='MultiViewPipeline', sequential=True, n_images=6, n_times=4, transforms=[
                dict(type='LoadImageFromFile', file_client_args=dict(backend='disk'))
            ]),
            dict(type='LoadPointsFromFile', dummy=True, coord_type='LIDAR', load_dim=5, use_dim=5),
            dict(type='RandomAugImageMultiViewImage', data_config=data_config, is_train=False),
            dict(type='KittiSetOrigin', point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(type='NormalizeMultiviewImage',
                 mean=[123.675, 116.28, 103.53],
                 std=[58.395, 57.12, 57.375],
                 to_rgb=True),
            dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
            dict(type='Collect3D', keys=['img'])
        ],
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        with_box2d=True,
        box_type_3d='LiDAR',
        ann_file='data/nuscenes/nuscenes_infos_val_4d_interval3_max60.pkl',
        load_interval=1,
        sequential=True,
        n_times=4,
        train_adj_ids=[1, 3, 5],
        speed_mode='abs_velo',
        max_interval=10,
        min_interval=0,
        fix_direction=True,
        test_adj='prev',
        test_adj_ids=[1, 3, 5],
        test_time_id=None,
    )

    # ----------------------------
    # 2. 构建数据集
    # ----------------------------
    dataset = build_dataset(dataset_cfg)
    print("Dataset type:", type(dataset))
    print("Dataset length:", len(dataset))
    print("Class names:", dataset.CLASSES)

    # ----------------------------
    # 3. 随机取一个 sample
    # ----------------------------
    idx = random.randint(0, len(dataset) - 1)
    sample = dataset[idx]
    print(f"Sample {idx} keys:", sample.keys())
    print_dict_tree(sample['img_metas'].data)

    if 'img' in sample:
        imgs = [img.permute(1, 2, 0).cpu().numpy() for img in sample['img'].data]
        show_images(imgs, title=f"Sample {idx} Multiview Images")



def print_dict_tree(data, indent=0, prefix='', processed_keys=None):
    """
    递归打印字典的树形结构，处理键名前4字符相同的情况
    :param data: 要处理的数据(字典或列表)
    :param indent: 当前缩进级别
    :param prefix: 连接线前缀
    :param processed_keys: 已处理的键名前缀集合
    """
    if processed_keys is None:
        processed_keys = set()

    if isinstance(data, dict):
        keys = list(data.keys())
        for i, key in enumerate(keys):
            is_last = (i == len(keys) - 1)
            connector = '└── ' if is_last else '├── '

            # 获取当前键的前4个字符
            key_prefix = str(key)[:4]

            # 检查是否需要跳过递归
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
        # 不是 dict/list，说明是值
        if isinstance(data, np.ndarray):
            # numpy 数组就打印 shape
            print(prefix + f'└── ndarray shape={data.shape}')
        else:
            # 其他类型就打印类型和值
            try:
                length = len(data)
            except Exception:
                length = None
            if length is not None:
                print(prefix + f'└── type={type(data).__name__}, len={length}')
            else:
                print(prefix + f'└── type={type(data).__name__}, value={data}')


if __name__ == "__main__":
    main()

'''
打印结果如下：
├── filename
│   └── type=str, len=104
├── ori_shape
│   └── type=tuple, len=3
├── img_shape
│   ├── [12]
│   │   └── type=tuple, len=3
├── lidar2img
│   ├── extrinsic
│   │   ├── [12]
│   │   │   └── ndarray shape=(4, 4)
│   ├── intrinsic
│   │   └── ndarray shape=(4, 4)
│   ├── lidar2img_aug
│   ├── lidar2img_extra
│   └── origin
│       └── ndarray shape=(3,)
├── box_mode_3d
│   └── type=Box3DMode, value=0
├── box_type_3d
├── img_norm_cfg
├── sample_idx
│   └── type=str, len=32
└── img_info
'''