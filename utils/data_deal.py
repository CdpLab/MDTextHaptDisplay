import os
import pandas as pd
from tqdm import tqdm  # 引入tqdm库

# 设置源目录和目标目录，使用双反斜杠
source_dir = "C:\\Users\\dying\\Desktop\\笔试装置\\data"
target_dir = "C:\\Users\\dying\\Desktop\\data\\multidata_small"

# 确保目标目录存在
os.makedirs(target_dir, exist_ok=True)

# 检查源目录是否存在
if not os.path.exists(source_dir):
    print(f"源目录不存在: {source_dir}")
else:
    # 获取源目录中的所有CSV文件
    csv_files = [f for f in os.listdir(source_dir) if f.endswith(".csv")]

    # 使用tqdm显示进度条
    for filename in tqdm(csv_files, desc="Processing CSV files", unit="file"):
        # 构造完整的文件路径
        source_file = os.path.join(source_dir, filename)

        # 读取CSV文件
        try:
            df = pd.read_csv(source_file)
        except FileNotFoundError:
            print(f"文件未找到: {source_file}")
            continue

        # 检查是否包含所需的列
        if 'speed' in df.columns and 'force_z' in df.columns and 'accel' in df.columns and 'friction' in df.columns:
            # 重命名列 force_z 为 force
            df.rename(columns={'force_z': 'force'}, inplace=True)

            # 选择需要的列
            df = df[['speed', 'force', 'accel', 'friction']]

            # 构造目标文件的路径
            target_file = os.path.join(target_dir, filename)

            # 保存修改后的DataFrame到新的CSV文件
            df.to_csv(target_file, index=False)

