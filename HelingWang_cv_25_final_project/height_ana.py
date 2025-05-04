# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:48:19 2025

@author: 81265
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# 你的 `.mat` 文件存放的路径
folder_path = r"E:\research_23_pinn\new_data\data.cresis.ku.edu\data\temp\internal_layers\NASA_OIB_test_files\image_files\Dataset\snow\SR_Dataset_v1\train_data"

# 存储所有文件的 20 层截止值
layer_20_end_values = []

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
                    # 提取第 20 层的值（索引从 0 开始，所以取 index=19）
                    layer_20_values = layers_vector[19, :]
                    
                    # 去除 NaN
                    valid_values = layer_20_values[~np.isnan(layer_20_values)]
                    
                    # 记录所有文件的 20 层截止值
                    if len(valid_values) > 0:
                        layer_20_end_values.extend(valid_values)

# 统计 20 层截止值的分布情况
if layer_20_end_values:
    layer_20_end_values = np.array(layer_20_end_values)
    
    # 计算统计信息
    min_val = np.min(layer_20_end_values)
    max_val = np.max(layer_20_end_values)
    mean_val = np.mean(layer_20_end_values)
    median_val = np.median(layer_20_end_values)

    # 打印统计结果
    print(f"20 层截止值统计：")
    print(f"- 最小值: {min_val:.2f}")
    print(f"- 最大值: {max_val:.2f}")
    print(f"- 均值: {mean_val:.2f}")
    print(f"- 中位数: {median_val:.2f}")

    # 绘制分布直方图
    plt.figure(figsize=(8, 5))
    plt.hist(layer_20_end_values, bins=30, color="blue", alpha=0.7)
    plt.xlabel("Layer 20 End Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Layer 20 End Values")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()
else:
    print("未找到任何符合条件的文件或 20 层数据无有效值")
