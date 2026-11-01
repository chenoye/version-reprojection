import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取 build 目录下的 CSV 数据
df = pd.read_csv('build/ekf_data.csv')

# 确保数据是数值型，并丢弃包含 NaN 的无效行
df = df.apply(pd.to_numeric, errors='coerce').dropna()

# ================= 数据平滑 =================
window_size = 7
for col in ['cx', 'cy', 'cz']:
    df[col] = df[col].rolling(window=window_size, min_periods=1).mean()

# Yaw 角度解卷绕
df['yaw_unwrapped'] = np.unwrap(df['yaw'].values)
df['yaw_unwrapped'] = df['yaw_unwrapped'].rolling(window=window_size, min_periods=1).mean()

# ================= 创建 2x2 画布 =================
plt.rcParams['axes.unicode_minus'] = False
fig, axs = plt.subplots(2, 2, figsize=(14, 9), facecolor='white')
fig.suptitle('EKF State Estimation (Smoothed)', fontsize=18, fontweight='bold')

# 图1：X 和 Y 位置
axs[0, 0].plot(df['cx'], label='X (cx)', color='#1f77b4', linewidth=1.5)
axs[0, 0].plot(df['cy'], label='Y (cy)', color='#ff7f0e', linewidth=1.5)
axs[0, 0].set_title('Horizontal Position', fontsize=14)
axs[0, 0].set_ylabel('Distance (m)', fontsize=12)
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle='--', alpha=0.6)

# 图2：Z 深度
axs[0, 1].plot(df['cz'], label='Z (cz)', color='#2ca02c', linewidth=1.5)
axs[0, 1].set_title('Depth (Z-axis)', fontsize=14)
axs[0, 1].set_ylabel('Distance (m)', fontsize=12)
axs[0, 1].legend()
axs[0, 1].grid(True, linestyle='--', alpha=0.6)

# 图3：Yaw 偏航角
axs[1, 0].plot(df['yaw_unwrapped'], label='Yaw', color='#d62728', linewidth=1.5)
axs[1, 0].set_title('Yaw Angle', fontsize=14)
axs[1, 0].set_ylabel('Angle (rad)', fontsize=12)
axs[1, 0].legend()
axs[1, 0].grid(True, linestyle='--', alpha=0.6)

# 图4：速度
axs[1, 1].plot(df['vx'], label='Vx', color='#9467bd', linewidth=1)
axs[1, 1].plot(df['vy'], label='Vy', color='#8c564b', linewidth=1)
axs[1, 1].plot(df['vz'], label='Vz', color='#e377c2', linewidth=1)
axs[1, 1].set_title('Velocity', fontsize=14)
axs[1, 1].set_ylabel('Speed (m/s)', fontsize=12)
axs[1, 1].legend()
axs[1, 1].grid(True, linestyle='--', alpha=0.6)

# 隐藏横坐标数字
for ax in axs.flat:
    ax.set_xlabel('Frame', fontsize=10)
    ax.tick_params(axis='x', labelbottom=False) 

# ================= 保存图片 =================
plt.tight_layout()
# 图片会生成在根目录下
plt.savefig('ekf_curves_pro.png', dpi=300, bbox_inches='tight')
print("✅ 成功！已生成图表: ekf_curves_pro.png")