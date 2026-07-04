import matplotlib.pyplot as plt
import numpy as np

# 1. 准备数据
# 定义 X 轴的组标签
labels = ['2', '5', '10', '20', '50']
# 定义每个组中三种方法的数据
# 我根据您图片中的大致高度估算了这些值
uaws_data = [88.37, 92.96, 95.22, 96.45, 97.85]
ppo_data = [87.89, 92.53, 93.66,94.82, 97.30]
a3c_data = [86.93, 91.97, 94.13, 96.26, 97.65]

# 2. 设置绘图参数
# 设置每个柱子的宽度
bar_width = 0.25
# 生成 X 轴的位置
x = np.arange(len(labels))

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 6))

# 3. 绘制柱状图
# 绘制  的柱状图
rects1 = ax.bar(x - bar_width, uaws_data, bar_width, 
                label='UAWS', 
                hatch='xx',       # 使用右斜线填充
                facecolor='white', # 设置柱子表面为白色
                edgecolor='black') # 设置边框为黑色

# 绘制  的柱状图
rects2 = ax.bar(x, ppo_data, bar_width, 
                label='HAC-PPO', 
                hatch='//',       # 使用交叉线填充
                facecolor='white', 
                edgecolor='black')

# 绘制  的柱状图
rects3 = ax.bar(x + bar_width, a3c_data, bar_width, 
                label='MCWS-A3C', 
                hatch='\\\\',     # 使用左斜线填充 (需要两个反斜杠进行转义)
                facecolor='white', 
                edgecolor='black')

# 4. 美化图表
# 设置 Y 轴标签
ax.set_ylabel('Utilization(%)', fontsize=14)
# 设置 X 轴标签
ax.set_xlabel('WorkflowNum', fontsize=14)

# 设置 X 轴的刻度标签
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)

# 设置 Y 轴的范围
ax.set_ylim(80, 102)

# 显示图例
ax.legend(loc='upper right')

# 移除顶部和右侧的边框线，使其更像示例图片
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整布局以防止标签重叠
fig.tight_layout()

# 5. 显示图表
# plt.show()
plt.savefig('kexue-resourceratio-wfnum.png', dpi=300, bbox_inches='tight')

# 可选：打印一条消息确认保存成功
print("图片已成功保存为 kexue-resourceratio-wfnum.png")