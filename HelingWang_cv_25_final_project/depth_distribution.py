# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 08:42:58 2025

@author: 81265
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 你的 `.mat` 文件存放的文件夹路径
folder_path = r"E:\research_23_pinn\new_data\data.cresis.ku.edu\data\temp\internal_layers\NASA_OIB_test_files\image_files\Dataset\snow\SR_Dataset_v1\train_data"

# 统计深度的列表
depth_counts = []

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

                # 记录深度
                depth_counts.append(depth)

# 统计不同深度的文件数
depth_counter = Counter(depth_counts)
depth_values, file_counts = zip(*sorted(depth_counter.items()))  # 按深度排序

# 计算累计比例
total_files = sum(file_counts)
file_ratios = [count / total_files for count in file_counts]
cumulative_ratios = np.cumsum(file_ratios)  # 累计比例

# 绘制图 1：文件比例 vs 深度
plt.figure(figsize=(8, 5))
plt.bar(depth_values, file_ratios, color='blue', alpha=0.7)
plt.xlabel("Depth")
plt.ylabel("File Ratio")
plt.title("Distribution of Files by Depth")
plt.xticks(depth_values)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# 绘制图 2：累计比例 vs 深度
plt.figure(figsize=(8, 5))
plt.bar([f">={d}" for d in depth_values], cumulative_ratios, color='red', alpha=0.7)
plt.xlabel("Depth (Cumulative)")
plt.ylabel("File Ratio")
plt.title("Cumulative Distribution of Files by Depth")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()

# 打印统计结果
print("Depth Distribution:")
for depth, count in depth_counter.items():
    print(f"Depth {depth}: {count} files ({count/total_files:.2%})")
# 打印总文件数
print(f"\nTotal number of .mat files processed: {total_files}")
