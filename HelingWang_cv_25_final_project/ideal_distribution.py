# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:12:21 2025

@author: 81265
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 你的 `.mat` 文件存放的文件夹路径
folder_path = r"E:\research_23_pinn\new_data\data.cresis.ku.edu\data\temp\internal_layers\NASA_OIB_test_files\image_files\Dataset\snow\SR_Dataset_v1\train_data"

# 统计深度 ≥ 20 的文件
files_with_large_depth = []

# 遍历文件夹中的所有 .mat 文件
for file_name in os.listdir(folder_path):
    if file_name.endswith(".mat"):  # 确保是 .mat 文件
        file_path = os.path.join(folder_path, file_name)
        
        # 读取 .mat 文件
        with h5py.File(file_path, "r") as f:
            if "layers_vector" in f:
                layers_vector = np.array(f["layers_vector"]).T  # 转置
                
                # 计算无 NaN 的最大深度
                row_nan_mask = np.any(np.isnan(layers_vector), axis=1)
                start_depth = None

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
                    files_with_large_depth.append(file_path)

# 统计 layers_bin_bitmap（转置）的深度
bin_bitmap_depths = []

for file_path in files_with_large_depth:
    with h5py.File(file_path, "r") as f:
        if "layers_bin_bitmap" in f:
            layers_bin_bitmap = np.array(f["layers_bin_bitmap"])
            transposed_shape = layers_bin_bitmap.T.shape  # 获取转置后的形状
            
            # 找到不是 256 的维度作为深度
            depth_dim = [dim for dim in transposed_shape if dim != 256]
            
            if depth_dim:
                bin_bitmap_depths.append(depth_dim[0])  # 记录深度

# 设定分档区间（bins）
bins = [799] + list(range(800, 1600, 100)) + [1501]  # 修正 bins 使其与 labels 对齐
bin_labels = [f"≤800"] + [f"{i}-{i+99}" for i in range(801, 1500, 100)] + [">1500"]

# 计算每个深度落入哪个区间
bin_counts, _ = np.histogram(bin_bitmap_depths, bins=bins)  # 修正长度匹配

# 绘制柱状图
plt.figure(figsize=(10, 5))
plt.bar(bin_labels, bin_counts, color="blue", alpha=0.7)
plt.xlabel("Layers_bin_bitmap Depth Ranges")
plt.ylabel("Number of Files")
plt.title("Distribution of Layers_bin_bitmap Depth (Transposed)")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# 打印统计结果
print("Layers_bin_bitmap (transposed) depth distribution:")
for label, count in zip(bin_labels, bin_counts):
    print(f"Depth {label}: {count} files")
