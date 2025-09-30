import torch
from mmcv import Config
from mmdet3d.models import build_detector
import pickle
from mmcv.parallel import DataContainer
import numpy as np


def move_to_device(data, device):
    """
    递归地把所有 tensor 移到 device
    """
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, np.ndarray):
        # numpy数组 -> tensor后再to(device)，如果你需要
        return torch.from_numpy(data).to(device)
    elif isinstance(data, DataContainer):
        return move_to_device(data.data[0], device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [move_to_device(v, device) for v in data]
    else:
        return data

def unwrap_dc(batch, device):
    """
    解开 batch 中的 DataContainer 并递归移到 device
    """
    return {k: move_to_device(v, device) for k, v in batch.items()}

def load_one_batch():
    with open("./myscripts/test_dataset/first_batch.pkl", 'rb') as f:
        batch_data = pickle.load(f)
    return batch_data

def main():
    # 1. 载入你的配置
    config_path = 'configs/fastbev/exp/paper/fastbev_r50_v4_ms_20cbgs_s256x704_v200x200x6_c256_d6_f4.py'
    cfg = Config.fromfile(config_path)
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0

    # 2. 创模型 + 移到CPU
    device = torch.device('cuda:0')
    model = build_detector(cfg.model,
                           train_cfg=cfg.get('train_cfg'),
                           test_cfg=cfg.get('test_cfg'))
    model = model.to(device)
    model.train()

    # 3. 载入 + unwrap 数据
    batch_data = load_one_batch()
    batch_data_unwrapped = unwrap_dc(batch_data, device)  # 全部移到CPU

    # 4. 前向+loss
    losses = model(return_loss=True, **batch_data_unwrapped)
    print("=== Fake training loss ===")
    for k, v in losses.items():
        try:
            print(f"{k}: {v.item()}")
        except:
            print(f"{k}: {v}")

    # 5. 验证 /推理
    # model.eval()
    # with torch.no_grad():
    #     outputs = model(return_loss=False, **batch_data_unwrapped)
    # print("=== Fake eval output ===")
    # print(outputs)

if __name__ == '__main__':
    main()
