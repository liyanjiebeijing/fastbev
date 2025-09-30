#!/bin/bash

# 最大并行下载数
MAX_JOBS=10

# # 下载 nuScenes v1.0 trainval 相机数据（共 10 个分片）
# for i in $(seq -w 1 10); do
#     url="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval${i}_blobs_camera.tgz"
#     echo "Starting download: $url"
#     wget -c $url &   # 后台运行

#     # 控制最大并发数
#     if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
#         wait -n  # 等待任意一个任务完成后再继续
#     fi
# done

# # 等待所有任务结束
# wait
# echo "✅ All downloads completed!"

# for i in $(seq -w 1 10); do
#     tar -xvzf v1.0-trainval${i}_blobs_camera.tgz
# done



# # 下载 nuScenes v1.0 trainval lidar数据（共 10 个分片）
# for i in $(seq -w 1 10); do
#     url="https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval${i}_blobs_lidar.tgz"

#     echo "Starting download: $url"
#     wget -c $url &   # 后台运行

#     # 控制最大并发数
#     if [[ $(jobs -r -p | wc -l) -ge $MAX_JOBS ]]; then
#         wait -n  # 等待任意一个任务完成后再继续
#     fi
# done

# # 等待所有任务结束
# wait
# echo "✅ All downloads completed!"

for i in $(seq -w 1 10); do
    tar -xvzf v1.0-trainval${i}_blobs_lidar.tgz
done



