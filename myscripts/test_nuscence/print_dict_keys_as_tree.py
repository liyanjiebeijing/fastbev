

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

# # 示例用法
# sample_data = {
#     'user123': {'name': 'Alice', 'age': 25},
#     'user456': {'name': 'Bob', 'age': 30},
#     'user789': {'name': 'Charlie', 'age': 35},  # 前4字符相同，不递归显示
#     'product1': {'price': 10, 'stock': 100},
#     'product2': {'price': 20, 'stock': 50},     # 前4字符相同，不递归显示
#     'order001': {'items': ['book', 'pen']},
#     'order002': {'items': ['pencil', 'eraser', 'knife']}  # 前4字符相同，不递归显示
# }
# print_dict_tree(sample_data)


import mmcv
data = mmcv.load("data/nuscenes/nuscenes_infos_train_4d_interval3_max60.pkl")
print_dict_tree(data)

'''
打印结果如下：
├── infos
│   ├── [14065]
│   │   ├── lidar_path
│   │   ├── token
│   │   ├── sweeps
│   │   │   ├── [5]
│   │   │   │   ├── data_path
│   │   │   │   ├── type
│   │   │   │   ├── sample_data_token
│   │   │   │   ├── sensor2ego_translation
│   │   │   │   │   ├── [1]
│   │   │   │   ├── sensor2ego_rotation
│   │   │   │   ├── ego2global_translation
│   │   │   │   │   ├── [1]
│   │   │   │   ├── ego2global_rotation
│   │   │   │   ├── timestamp
│   │   │   │   ├── sensor2lidar_rotation
│   │   │   │   └── sensor2lidar_translation
│   │   ├── cams
│   │   │   ├── CAM_FRONT
│   │   │   │   ├── data_path
│   │   │   │   ├── type
│   │   │   │   ├── sample_data_token
│   │   │   │   ├── sensor2ego_translation
│   │   │   │   ├── sensor2ego_rotation
│   │   │   │   ├── ego2global_translation
│   │   │   │   ├── ego2global_rotation
│   │   │   │   ├── timestamp
│   │   │   │   ├── sensor2lidar_rotation
│   │   │   │   ├── sensor2lidar_translation
│   │   │   │   └── cam_intrinsic
│   │   │   ├── CAM_FRONT_RIGHT
│   │   │   ├── CAM_FRONT_LEFT
│   │   │   ├── CAM_BACK
│   │   │   ├── CAM_BACK_LEFT
│   │   │   └── CAM_BACK_RIGHT
│   │   ├── lidar2ego_translation
│   │   ├── lidar2ego_rotation
│   │   ├── ego2global_translation
│   │   ├── ego2global_rotation
│   │   ├── timestamp
│   │   ├── gt_boxes
│   │   ├── gt_names
│   │   ├── gt_velocity
│   │   ├── num_lidar_pts
│   │   ├── num_radar_pts
│   │   ├── valid_flag
│   │   ├── next
│   │   │   ├── [10]
│   │   │   │   ├── timestamp
│   │   │   │   ├── cams
│   │   │   │   ├── ego2global_translation
│   │   │   │   └── ego2global_rotation
│   │   ├── prev
│   │   │   ├── [4]
│   │   │   │   ├── timestamp
│   │   │   │   ├── cams
│   │   │   │   ├── ego2global_translation
│   │   │   │   └── ego2global_rotation
│   │   └── velo
└── metadata

'''