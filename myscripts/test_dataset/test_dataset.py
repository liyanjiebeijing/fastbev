
from mmcv import Config
from mmdet3d.datasets import build_dataset
import mmcv
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取配置文件（你的 Fast-BEV config）
cfg = Config.fromfile('configs/fastbev/exp/paper/fastbev_m0_r18_s256x704_v200x200x4_c192_d2_f4.py')

# 2. 用配置构建 dataset
dataset = build_dataset(cfg.data.train)

# 3. 取一个样本
data = dataset[0]   # 或者 for data in dataset: ...

print("keys:", data.keys())
print("img shape:", data['img'].data.shape if 'img' in data else None)
