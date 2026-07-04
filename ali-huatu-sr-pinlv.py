import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
# X 轴数据 (Failure Probability)
x = ['0.1', '0.3', '0.5', '0.7', '1']
# x = np.array(x_labels)

# Y 轴数据 (根据图片目测估算)
# GRWS (圆形标记，虚线，数值较低且大部分为负)
grws_data = [62, 54, 54, 56, 59]

# DQN (菱形标记，点划线，波动剧烈)
dqn_data = [30, 12, 18, 19, 19]

# ETTS (实心点/小标记，实线，最后飙升)
etts_data = [62, 45, 48, 51, 55]

# 2. 设置绘图参数
fig, ax = plt.subplots(figsize=(6, 5)) # 调整画布比例以接近原图

# 3. 绘制折线图
# 绘制 GRWS
ax.plot(x, grws_data, 
        label='UAWS', 
        color='black',          # 线条颜色：黑
        linestyle='-',         # 线型：虚线
        marker='o',             # 标记：圆圈
        markersize=5)           # 标记大小

# 绘制 DQN
ax.plot(x, dqn_data, 
        label='HAC-PPO', 
        color='black',          # 线条颜色：黑
        linestyle='-.',         # 线型：点划线
        marker='s',             # 标记：
        markersize=5)

# 绘制 ETTS
ax.plot(x, etts_data, 
        label='MCWS-A3C', 
        color='black',          # 线条颜色：黑
        linestyle=':',          # 线型：实线
        marker='^',             # 标记：点 (或者用 '*' 星号)
        markersize=8)           # 点标记通常需要大一点才明显

# 4. 美化图表
# 设置坐标轴标签
ax.set_xlabel('Arrival Rate', fontsize=14)
ax.set_ylabel('Success Ratio(%)', fontsize=14)

# 设置 X 轴刻度
ax.set_xticks(x)
# 如果希望显示得更精确（比如像图片里的间隔），可以保持默认或手动指定
# ax.set_xticklabels(['0.02', '0.04', '0.06', '0.08', '0.10'])

# 设置 Y 轴范围 (根据图片约为 -70 到 180)
ax.set_ylim(0, 100)

# 去除顶部和右侧边框 (Spines)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 设置图例
# frameon=False 去除图例边框，loc设置位置
ax.legend(loc='upper right', frameon=False, fontsize=9)

# 调整布局
fig.tight_layout()

# 5. 保存与显示
plt.savefig('ali-sr-wfnum.png', dpi=300, bbox_inches='tight')
plt.show()

print("图片已成功保存为 ali-sr-wfnum.png")