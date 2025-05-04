# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:22:43 2025

@author: 81265
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
# 指定 `.mat` 文件路径
addr = r"E:\desktop\cse498\final_project\processed_data_part3.mat"
k = 0  # 选择要读取的数据索引

with h5py.File(addr, "r") as f:
    print(f["LayersVector"].shape)
    new_data = f["Data"][:50,:,:]
    new_segment_bitmap = f["SegmentBitmap"][:50,:,:]
    new_layers_vector = f["LayersVector"][:50,:,:]
    data = np.array(f["Data"])[k]
    segment_bitmap = np.array(f["SegmentBitmap"])[k]
    layers_vector = np.array(f["LayersVector"])[k]


print(data.shape,segment_bitmap.shape,layers_vector.shape)
print(layers_vector)

# 可视化 Data
plt.figure(figsize=(4,8))
plt.imshow(data, cmap='viridis', interpolation='none')
plt.colorbar(label="Data Value")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.title("Data")
plt.tight_layout()
plt.show()
#layer_x = np.arange(layers_vector.shape[1])
#for i in range(layers_vector.shape[0]):
#    if i <5 :
#        plt.scatter(layer_x, layers_vector[i], color='blue', s=5, label=f"Layer {i}" if i == 0 else "")
# 可视化 SegmentBitmap
plt.figure(figsize=(4, 8))
plt.imshow(segment_bitmap, cmap='gray', interpolation='none')
plt.colorbar(label="Segment Bitmap Value")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.title("SegmentBitmap")

# 叠加 Layer Vector
#layer_x = np.arange(layers_vector.shape[1])
#for i in range(layers_vector.shape[0]):
#    plt.scatter(layer_x, layers_vector[i], color='red', s=5, label=f"Layer {i}" if i == 0 else "")

plt.legend()
plt.tight_layout()
plt.show()

fig = plt.figure(figsize = (4,8))
#plt.subplot((121))
empty = np.ones_like(data,dtype=np.uint8)*255
plt.imshow(empty,cmap = 'gray', vmin=0, vmax=255)
#plt.imshow(image,cmap = 'gray')
#plt.axis('off')

for i in range(20):
    if i <5 :
        plt.plot(layers_vector[i],color = 'red')
    else:
        plt.plot(layers_vector[i],color = 'blue')
#plt.subplot((1,2,2))
#plt.imshow(mask)

plt.show()


'''
output_path = "small_subset.mat"

with h5py.File(output_path, "w") as f_out:
    f_out.create_dataset("Data", data=new_data)
    f_out.create_dataset("SegmentBitmap", data=new_segment_bitmap)
    f_out.create_dataset("LayersVector", data=new_layers_vector)

print(f"小数据集保存成功：{output_path}")
'''