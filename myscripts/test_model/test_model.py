from mmcv import Config
from mmdet3d.models import build_detector

# 读取配置文件
cfg = Config.fromfile('configs/fastbev/exp/paper/fastbev_r50_v4_ms_20cbgs_s256x704_v200x200x6_c256_d6_f4.py')

# 修改 cfg 让其适合假数据（比如 batch size）
cfg.data.samples_per_gpu = 1
cfg.data.workers_per_gpu = 0

# 创建模型
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

# 放到 GPU/CPU
model = model.cuda()
model.eval()
