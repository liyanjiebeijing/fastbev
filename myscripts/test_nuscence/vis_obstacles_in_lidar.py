
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


nuscence_root_dir="/mnt/share_disk/liyanjie/project/Fast-BEV/data/nuscenes_v1.0_mini"
nusc = NuScenes(version='v1.0-mini', dataroot=nuscence_root_dir, verbose=True)


def draw_lidar(save_path="lidar_vis.png"):
    # --- 1. 获取 LiDAR 数据和标注框 ---
    my_sample = nusc.sample[20]
    lidar_token = my_sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_filepath = nusc.get_sample_data_path(lidar_token)

    # 加载点云，参数 `points_dim=5` 表示加载 (x, y, z, intensity, ring_index)
    pc = LidarPointCloud.from_file(lidar_filepath)
    points = pc.points[:3, :].T # 只取 x, y, z 坐标

    # 获取在此 LiDAR 坐标系下的所有标注框
    # get_sample_data 会自动完成 世界坐标系 -> LiDAR坐标系 的转换
    _, boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=1)


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制点云
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=0.2, c="gray")

    # 绘制标注框
    for box in boxes:
        corners = box.corners().T  # (8, 3)
        ax.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=0.1, c="r")

    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存成功: {save_path}")


    # --- 2. 使用 Open3D 可视化 ---
    # # 创建 Open3D 点云对象
    # o3d_pc = o3d.geometry.PointCloud()
    # o3d_pc.points = o3d.utility.Vector3dVector(points)

    # # 创建 Open3D 边界框对象
    # o3d_boxes = []
    # for box in boxes:
    #     # devkit 的 box 中心点在底部，open3d 在中部，做一个小的修正
    #     center = box.center - np.array([0, 0, box.wlh[2] / 2])
    #     o3d_box = o3d.geometry.OrientedBoundingBox(center, box.rotation_matrix, box.wlh)
    #     o3d_box.color = (1, 0, 0) # 设置为红色
    #     o3d_boxes.append(o3d_box)

    # --- 3. 离屏渲染并保存 ---
    # vis = o3d.visualization.Visualizer()
    # vis.create_window(visible=False)  # 不显示窗口
    # vis.add_geometry(o3d_pc)
    # for b in o3d_boxes:
    #     vis.add_geometry(b)
    # vis.poll_events()
    # vis.update_renderer()
    # vis.capture_screen_image(save_path)
    # vis.destroy_window()
    # print(f"Saved to {save_path}")


    # ✅ 使用离屏渲染器
    # vis = o3d.visualization.rendering.OffscreenRenderer(1024, 768)
    # mat = o3d.visualization.rendering.MaterialRecord()
    # mat.shader = "defaultUnlit"

    # vis.scene.add_geometry("pc", o3d_pc, mat)
    # for i, b in enumerate(o3d_boxes):
    #     vis.scene.add_geometry(f"box{i}", b, mat)

    # # 设置相机参数（简单点：自动放缩）
    # vis.setup_camera(60.0, o3d_pc.get_axis_aligned_bounding_box(), o3d_pc.get_center())

    # # 渲染到 numpy
    # img = vis.render_to_image()
    # o3d.io.write_image(save_path, img)
    # print(f"保存成功: {save_path}")


def sample_line(p1, p2, num=10):
    """在两点之间均匀采样 num 个点，包括端点"""
    return np.linspace(p1, p2, num=num)

def save_lidar_with_colored_boxes(sample_index=20,
                                   pcd_path="./lidar_with_boxes.pcd",
                                   txt_path="./lidar_with_boxes.txt",
                                   box_color=(0, 1, 0)):
    # 1. 获取 LiDAR 数据
    my_sample = nusc.sample[sample_index]
    lidar_token = my_sample['data']['LIDAR_TOP']
    lidar_filepath = nusc.get_sample_data_path(lidar_token)

    # 2. 加载点云 (x, y, z)
    pc = LidarPointCloud.from_file(lidar_filepath)
    points = pc.points[:3, :].T  # shape: (N, 3)

    # 给原始点云默认灰色
    points_colors = np.tile(np.array([[0.5, 0.5, 0.5]]), (points.shape[0], 1))

    # 3. 获取 LiDAR 坐标系下的标注框
    _, boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=1)

    line_points_list = []
    line_colors_list = []

    # 4. 对每个 box，采样边线点并上色
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
    # draw_lidar("lidar_vis.png")
    save_lidar_with_colored_boxes()