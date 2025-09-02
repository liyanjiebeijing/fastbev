from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
from nuscenes.utils.data_classes import LidarPointCloud, Box
from pyquaternion import Quaternion
import numpy as np


nuscence_root_dir="/mnt/share_disk/liyanjie/project/Fast-BEV/data/nuscenes_v1.0_mini"
nusc = NuScenes(version='v1.0-mini', dataroot=nuscence_root_dir, verbose=True)


def sample_line(p1, p2, num=10):
    """在两点之间均匀采样 num 个点，包括端点"""
    return np.linspace(p1, p2, num=num)

def vis_point_cloud(sample_index=20,
                    pcd_path="./lidar_with_boxes.pcd",
                    txt_path="./lidar_with_boxes.txt",
                    box_color=(0, 1, 0)):
    # --- 1. 初始化和加载数据 ---
    my_sample = nusc.sample[sample_index]
    lidar_token = my_sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    cs_record = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    lidar_filepath = nusc.get_sample_data_path(lidar_token)

    # --- 2. 加载点云 (点云本身就在 LiDAR 坐标系下) ---
    pc = LidarPointCloud.from_file(lidar_filepath)
    points = pc.points[:3, :].T
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(points)

    # 给原始点云默认灰色
    points_colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (points.shape[0], 1))

    # --- 3. 核心：手动转换所有标注框到 LiDAR 坐标系 ---
    boxes = []
    for ann_token in my_sample['anns']:
        ann_record = nusc.get('sample_annotation', ann_token)
        box = Box(ann_record['translation'], ann_record['size'], Quaternion(ann_record['rotation']))

        # a) 将 Box 从世界坐标系转换到自车坐标系
        box.translate(-np.array(pose_record['translation']))
        box.rotate(Quaternion(pose_record['rotation']).inverse)

        # b) 将 Box 从自车坐标系转换到 LiDAR 坐标系
        box.translate(-np.array(cs_record['translation']))
        box.rotate(Quaternion(cs_record['rotation']).inverse)

        boxes.append(box)


    # 4. 对每个 box，采样边线点并上色
    line_points_list = []
    line_colors_list = []
    for box in boxes:
        center = box.center - np.array([0, 0, box.wlh[2] / 2])
        corners = box.corners().T  # shape: (8, 3)

        # box 12 条边的顶点索引
        connections = [
            [0, 1], [1, 2], [2, 3], [3, 0],   # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],   # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]    # 竖线
        ]
        for i1, i2 in connections:
            line_points = sample_line(corners[i1], corners[i2], num=10)
            line_points_list.append(line_points)
            # 给框线点上指定颜色
            line_colors_list.append(np.tile(np.array([box_color]), (line_points.shape[0], 1)))


    if line_points_list:
        line_points = np.vstack(line_points_list)
        line_colors = np.vstack(line_colors_list)
        all_points = np.vstack([points, line_points])
        all_colors = np.vstack([points_colors, line_colors])
    else:
        all_points = points
        all_colors = points_colors


    # 5. 保存为 PCD
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(all_points)
    o3d_pc.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(pcd_path, o3d_pc)
    print(f"Saved PCD to {pcd_path}")

    # 6. 保存为 TXT (不含颜色)
    np.savetxt(txt_path, all_points, fmt="%.6f")
    print(f"Saved TXT to {txt_path}")



if __name__ == '__main__':
    vis_point_cloud()