import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import hbar, m_p
import matplotlib.ticker as ticker
import pandas as pd

def tunneling_probability(E, V0, width):
    """计算质子隧穿概率"""
    if E < V0:
        k = np.sqrt(2 * m_p * (V0 - E)) / hbar
        return np.exp(-2 * k * width)
    else:
        return 1.0

# 参数设置
E = 0.5 * 1.602e-19       # 质子能量 (0.5 eV)
V0 = 1.0 * 1.602e-19      # 势垒高度 (1.0 eV)
widths = np.linspace(0.1, 2.0, 100) * 1e-10  # 势垒宽度 (0.1-2.0 Å)

# 计算并可视化
probs = [tunneling_probability(E, V0, w) for w in widths]

plt.figure(figsize=(10,6))
plt.plot(widths*1e10, np.array(probs)*100, 'b-', lw=2)  # 将概率转换为百分比
plt.xlabel("Barrier Width (Å)", fontsize=12)
plt.ylabel("Tunneling Probability (%)", fontsize=12)    # 修改y轴标签为百分比
plt.title("Proton Tunneling Probability vs Barrier Width", fontsize=14)
plt.ylim(0, 5)                                        # 设置y轴范围为0%到100%
plt.grid(ls='--', alpha=0.7)

# 设置y轴刻度为百分比格式
formatter = ticker.FormatStrFormatter('%1.1f')
plt.gca().yaxis.set_major_formatter(formatter)

# 将数据导出到Excel
data = {
    "Barrier Width (Å)": widths * 1e10,
    "Tunneling Probability (%)": np.array(probs) * 100
}
df = pd.DataFrame(data)
df.to_excel(r"way\to\your\path\tunneling_data.xlsx", index=False)


plt.show()
