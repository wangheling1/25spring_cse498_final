# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 18:14:10 2025

@author: 81265
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for resizing

# 你的 `.mat` 文件存放的文件夹路径
folder_path = r"E:\research_23_pinn\new_data\data.cresis.ku.edu\data\temp\internal_layers\NASA_OIB_test_files\image_files\Dataset\snow\SR_Dataset_v1\train_data"

# 统计 20 层的最大值和最终图片高度
layer_20_max_values = []
image_heights = []  # 存储所有文件的最终图片高度
files_with_large_depth = []  # 存储深度 ≥ 20 的文件路径
first_valid_file = None  # 记录第一个符合条件的文件

# 目标尺寸
target_height = 1000

# 存储所有数据的列表
all_data = []
all_segment_bitmaps = []
all_layers_vectors = []

# 设置分批存储的文件计数
batch_size = 600  # 每个文件最多存储 600 组数据
file_count = 1  # 计数器

# 遍历文件夹中的所有 .mat 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".mat"):  # 确保是 .mat 文件
        file_path = os.path.join(folder_path, file_name)
        
        # 读取 .mat 文件
        with h5py.File(file_path, "r") as f:
            if "layers_vector" in f and "layers_segment_bitmap" in f and "Data" in f:
                layers_vector = np.array(f["layers_vector"]).T  # 转置
                layers_segment_bitmap = np.array(f["layers_segment_bitmap"]).T  # 转置
                data = np.array(f["Data"]).T  # 转置
                
                # 计算无 NaN 的最大深度
                row_nan_mask = np.any(np.isnan(layers_vector), axis=1)
                start_depth = None
                depth = 0  # 确保 depth 一定有值

                for i, has_nan in enumerate(row_nan_mask):
                    if not has_nan:  # 该行无 NaN
                        if start_depth is None:
                            start_depth = i
                    else:  # 该行有 NaN
                        if start_depth is not None:
                            depth = i  # 计算深度
                            break
                else:
                    depth = len(row_nan_mask)  # 如果没有 NaN，则深度为最大行数

                # 只保留深度 ≥ 20 的文件
                if depth >= 20:
                    files_with_large_depth.append(file_path)  # 存储文件路径

                    # 计算 20 层的最大值
                    layer_20_values = layers_vector[19, :]
                    valid_layer_20_values = layer_20_values[~np.isnan(layer_20_values)]  # 去除 NaN
                    
                    if len(valid_layer_20_values) > 0:
                        layer_20_max = int(np.max(valid_layer_20_values))  # 取整
                        layer_20_max_values.append(layer_20_max)
                    else:
                        layer_20_max = layers_segment_bitmap.shape[0]  # 如果无有效值，使用最大图像尺寸
                    
                    # 计算截取范围
                    if layers_vector.shape[0] > 20:  # 如果有超过 20 层
                        layer_21_values = layers_vector[20, :]
                        valid_layer_21_values = layer_21_values[~np.isnan(layer_21_values)]  # 去除 NaN
                        if len(valid_layer_21_values) > 0:
                            layer_21_min = int(np.min(valid_layer_21_values))  # 计算 21 层最小值
                            crop_end = max(layer_21_min - 30, layer_20_max)
                        else:
                            crop_end = layer_20_max  # 如果 21 层无有效值，默认使用 20 层最大值
                    else:
                        crop_end = layers_segment_bitmap.shape[0]  # 只有 20 层时，保持原图

                    crop_end = min(crop_end, layers_segment_bitmap.shape[0])  # 确保不超过原图大小
                    image_heights.append(crop_end)  # 记录最终图片的高度

                    # 截取 layers_segment_bitmap 和 data
                    truncated_segment_bitmap = layers_segment_bitmap[:crop_end, :]
                    truncated_data = data[:crop_end, :]
                    
                    # 计算缩放比例
                    scale_factor = target_height / crop_end
                    
                    # 调整 layers_vector 的值（但不改变 shape）
                    scaled_layers_vector = layers_vector * scale_factor
                    
                    # 调整 layers_segment_bitmap 和 data 的尺寸
                    resized_segment_bitmap = cv2.resize(truncated_segment_bitmap, (truncated_segment_bitmap.shape[1], target_height), interpolation=cv2.INTER_LINEAR)
                    resized_data = cv2.resize(truncated_data, (truncated_data.shape[1], target_height), interpolation=cv2.INTER_LINEAR)
                    
                    # 存储数据
                    all_data.append(resized_data)
                    all_segment_bitmaps.append(resized_segment_bitmap)
                    all_layers_vectors.append(scaled_layers_vector[:20, :])

                    # 当数据达到 batch_size 时保存并清空列表
                    if len(all_data) >= batch_size:
                        output_file = f"processed_data_part{file_count}.mat"
                        with h5py.File(output_file, "w") as f:
                            f.create_dataset("Data", data=np.array(all_data, dtype=np.float32))
                            f.create_dataset("SegmentBitmap", data=np.array(all_segment_bitmaps, dtype=np.float32))
                            f.create_dataset("LayersVector", data=np.array(all_layers_vectors, dtype=np.float32))
                        print(f"数据已保存到 {output_file}")
                        
                        # 清空列表，准备存储下一批
                        all_data.clear()
                        all_segment_bitmaps.clear()
                        all_layers_vectors.clear()
                        file_count += 1

# 存储最后一批数据
if all_data:
    output_file = f"processed_data_part{file_count}.mat"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("Data", data=np.array(all_data, dtype=np.float32))
        f.create_dataset("SegmentBitmap", data=np.array(all_segment_bitmaps, dtype=np.float32))
        f.create_dataset("LayersVector", data=np.array(all_layers_vectors, dtype=np.float32))
    print(f"数据已保存到 {output_file}")

# **统计所有文件的最终图片高度分布**
if image_heights:
    plt.figure(figsize=(8, 5))
    plt.hist(image_heights, bins=20, color="blue", alpha=0.7)
    plt.xlabel("Image Height")
    plt.ylabel("Number of Files")
    plt.title("Distribution of Final Image Heights")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

