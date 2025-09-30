from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader
from torch.utils.data import DataLoader
from mmcv.parallel import DataContainer
import pickle
import torch
import numpy as np



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
        elif isinstance(inner_data, (list, dict)):
            print(prefix + f'└── DataContainer (type={type(inner_data).__name__})')
            # 递归打印内部数据
            print_dict_tree(inner_data, indent, prefix + '│   ', processed_keys)
        else:
            print(prefix + f'└── DataContainer value={inner_data}')
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




def main():
    # 1. 读取配置文件
    cfg = Config.fromfile(
        'configs/fastbev/exp/paper/fastbev_r50_v4_ms_20cbgs_s256x704_v200x200x6_c256_d6_f4.py'
    )

    # 2. 构建 dataset
    dataset = build_dataset(cfg.data.train)

    # 3. 用 DataLoader 包装
    # cfg.data.samples_per_gpu 里是 batch_size
    batch_size = cfg.data.get('samples_per_gpu', 4)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.get('samples_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 0),
        dist=False,
        shuffle=False
    )

    # 4. 遍历一个 batch
    for i, data_batch in enumerate(data_loader):
        print(f"第{i}个batch keys: {data_batch.keys()}")
        # 这里 data_batch['img'] 是 DataContainer，需要 .data[0] 取 tensor
        if 'img' in data_batch:
            imgs = data_batch['img'].data[0]  # shape: (B, num_cams, 3, H, W)
            print("img shape:", imgs.shape)
        # 保存第一个 batch
        if i == 0:
            with open('first_batch.pkl', 'wb') as f:
                pickle.dump(data_batch, f)
            print("已保存第一个 batch 到 first_batch.pkl")

            print_dict_tree(data_batch)

            break


if __name__ == "__main__":
    main()