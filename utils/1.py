import matplotlib

matplotlib.use('Agg')  # 设置为非交互式后端

import os
import matplotlib.pyplot as plt
import numpy as np

# 设置源目录和目标目录
csv_dir = "C:\\Users\\dying\\Desktop\\data\\multidata_small"
image_dir = "C:\\Users\\dying\\Desktop\\data\\image_small"

# 确保目标目录存在
os.makedirs(image_dir, exist_ok=True)

# 获取源目录中的所有CSV文件
csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]

# 遍历所有CSV文件
for csv_file in csv_files:
    # 构造PNG文件的路径
    image_file = os.path.join(image_dir, os.path.splitext(csv_file)[0] + ".png")

    # 创建一个纯黑色的图像，尺寸为 (10, 10)，色彩为黑色 (RGB)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.zeros((10, 10, 3)), cmap='gray', vmin=0, vmax=1)  # 使用全零的数组创建纯黑色图片
    ax.axis('off')  # 不显示坐标轴

    # 保存图片
    plt.savefig(image_file, bbox_inches='tight', pad_inches=0, dpi=300)

    # 关闭图像以释放内存
    plt.close(fig)

    print(f"已保存图片: {image_file}")
