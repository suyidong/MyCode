"""
seed test with SISSO
Author: suyi dong
"""

from TorchSisso import SissoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib as mpl

# 读取训练和测试数据
traindata = pd.read_excel(r"D:\NO.1\PROCESS\二审修改2025.8.25（ddl9.14）\dataset -train.xlsx", sheet_name=0)
testdata = pd.read_excel(r"D:\NO.1\PROCESS\二审修改2025.8.25（ddl9.14）\dataset - test.xlsx", sheet_name=0)

print("训练数据:")
print(traindata)
print("测试数据:")
print(testdata)

# 定义一元和二元运算符
operators = ['+', '-', '*', '/', '||']

# 存储结果的列表
results = []

# 设置随机种子范围
SEED_RANGE = 20  # 测试的种子数量

print(f"开始种子敏感性分析，将使用 {SEED_RANGE} 个不同的随机种子...")

# 使用原始的训练和测试数据作为基准
sm_original = SissoModel(traindata, operators, None, 2, 4, 10)
rmse_original, equation_original, r2_original, _ = sm_original.fit()
p_test_original, _ = sm_original.evaluate(equation_original, testdata)
r2_test_original = r2_score(testdata.Target, p_test_original)
rmse_original_test = np.sqrt(mean_squared_error(testdata.Target, p_test_original))

print(f"原始模型 (固定训练/测试集):")
print(f"训练R²: {r2_original:.4f}")
print(f"测试R²: {r2_test_original:.4f}")
print(f"方程: {equation_original}")
results.append({
            'seed': -1,
            'r2_train': r2_original,
            'r2_test': r2_test_original,
            'rmse_train': rmse_original,
            'rmse_test': rmse_original_test,
            'equation': equation_original
        })
# 合并训练和测试数据
fulldata = pd.concat([traindata, testdata], ignore_index=True)

for seed in range(SEED_RANGE):
    print(f"\n处理种子 {seed + 1}/{SEED_RANGE}")

    try:
        # 随机打乱数据
        shuffled_data = fulldata.sample(frac=1, random_state=seed).reset_index(drop=True)

        # 重新划分训练和测试集 (80/20)
        split_idx = int(0.8 * len(shuffled_data))
        train_data = shuffled_data.iloc[:split_idx]
        test_data = shuffled_data.iloc[split_idx:]

        print(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")

        # 创建SISSO模型对象
        sm = SissoModel(train_data, operators, None, 2, 4)

        # 运行SISSO算法
        rmse, equation, r2, _ = sm.fit()
        print(f"获得的方程: {equation}")

        # 评估模型
        p_test, _ = sm.evaluate(equation, test_data)

        # 计算指标
        r2_train = r2
        r2_test = r2_score(test_data.Target, p_test)
        rmse_train = rmse
        rmse_test = np.sqrt(mean_squared_error(test_data.Target, p_test))

        # 存储结果
        results.append({
            'seed': seed,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'equation': equation
        })

    except Exception as e:
        print(f"种子 {seed} 运行失败: {str(e)}")
        continue

# 检查是否有成功运行的结果
if not results:
    print("所有种子运行均失败，请检查数据和参数设置")
    exit()

# 分析结果
print("\n种子敏感性分析结果:")
print(f"成功运行次数: {len(results)}/{SEED_RANGE}")
print(f"平均训练R²: {np.mean([r['r2_train'] for r in results]):.4f} ± {np.std([r['r2_train'] for r in results]):.4f}")
print(f"平均测试R²: {np.mean([r['r2_test'] for r in results]):.4f} ± {np.std([r['r2_test'] for r in results]):.4f}")
print(
    f"平均训练RMSE: {np.mean([r['rmse_train'] for r in results]):.4f} ± {np.std([r['rmse_train'] for r in results]):.4f}")
print(
    f"平均测试RMSE: {np.mean([r['rmse_test'] for r in results]):.4f} ± {np.std([r['rmse_test'] for r in results]):.4f}")

# 检查方程的一致性
equations = [r['equation'] for r in results]
unique_equations = set(equations)
print(f"\n发现 {len(unique_equations)} 个不同的方程:")
for i, eq in enumerate(unique_equations):
    print(f"方程 {i + 1}: {eq}")
    count = equations.count(eq)
    print(f"出现次数: {count} ({count / len(results) * 100:.1f}%)")

# 将结果保存到Excel文件
results_df = pd.DataFrame(results)
# 重新排列列的顺序
results_df = results_df[['seed', 'r2_train', 'r2_test', 'rmse_train', 'rmse_test', 'equation']]

# 创建统计信息DataFrame
stats_data = {
    'Metric': ['训练R²', '测试R²', '训练RMSE', '测试RMSE'],
    'Mean': [
        np.mean([r['r2_train'] for r in results]),
        np.mean([r['r2_test'] for r in results]),
        np.mean([r['rmse_train'] for r in results]),
        np.mean([r['rmse_test'] for r in results])
    ],
    'Std': [
        np.std([r['r2_train'] for r in results]),
        np.std([r['r2_test'] for r in results]),
        np.std([r['rmse_train'] for r in results]),
        np.std([r['rmse_test'] for r in results])
    ],
    'Min': [
        np.min([r['r2_train'] for r in results]),
        np.min([r['r2_test'] for r in results]),
        np.min([r['rmse_train'] for r in results]),
        np.min([r['rmse_test'] for r in results])
    ],
    'Max': [
        np.max([r['r2_train'] for r in results]),
        np.max([r['r2_test'] for r in results]),
        np.max([r['rmse_train'] for r in results]),
        np.max([r['rmse_test'] for r in results])
    ]
}
stats_df = pd.DataFrame(stats_data)

# 创建方程频率DataFrame
eq_freq_data = []
for eq in unique_equations:
    count = equations.count(eq)
    eq_freq_data.append({
        'Equation': eq,
        'Frequency': count,
        'Percentage': f"{count / len(results) * 100:.1f}%"
    })
eq_freq_df = pd.DataFrame(eq_freq_data)

# 保存到Excel文件
with pd.ExcelWriter('seed_sensitivity_analysis_results.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='详细结果', index=False)
    stats_df.to_excel(writer, sheet_name='统计信息', index=False)
    eq_freq_df.to_excel(writer, sheet_name='方程频率', index=False)

print("\n结果已保存到 'seed_sensitivity_analysis_results.xlsx'")

# 可视化结果
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'figure.figsize': (12, 10),
    'axes.grid': True,
    'grid.alpha': 0.3
})

# 创建性能指标图表
if len(results) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # R²图表
    axes[0, 0].plot([r['seed'] for r in results], [r['r2_train'] for r in results], 'ro-', alpha=0.7, label='Train R²')
    axes[0, 0].plot([r['seed'] for r in results], [r['r2_test'] for r in results], 'bo-', alpha=0.7, label='Test R²')
    axes[0, 0].axhline(y=r2_original, color='g', linestyle='--', label='Original Train R²')
    axes[0, 0].axhline(y=r2_test_original, color='m', linestyle='--', label='Original Test R²')
    axes[0, 0].set_xlabel('Seed')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].legend()
    axes[0, 0].set_title('R² vs Seed')

    # RMSE图表
    axes[0, 1].plot([r['seed'] for r in results], [r['rmse_train'] for r in results], 'ro-', alpha=0.7,
                    label='Train RMSE')
    axes[0, 1].plot([r['seed'] for r in results], [r['rmse_test'] for r in results], 'bo-', alpha=0.7,
                    label='Test RMSE')
    axes[0, 1].set_xlabel('Seed')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].legend()
    axes[0, 1].set_title('RMSE vs Seed')

    # R²分布直方图
    axes[1, 0].hist([r['r2_train'] for r in results], bins=20, alpha=0.7, label='Train R²', color='red')
    axes[1, 0].hist([r['r2_test'] for r in results], bins=20, alpha=0.7, label='Test R²', color='blue')
    axes[1, 0].axvline(x=r2_original, color='g', linestyle='--', label='Original Train R²')
    axes[1, 0].axvline(x=r2_test_original, color='m', linestyle='--', label='Original Test R²')
    axes[1, 0].set_xlabel('R²')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].set_title('R² Distribution')

    # RMSE分布直方图
    axes[1, 1].hist([r['rmse_train'] for r in results], bins=20, alpha=0.7, label='Train RMSE', color='red')
    axes[1, 1].hist([r['rmse_test'] for r in results], bins=20, alpha=0.7, label='Test RMSE', color='blue')
    axes[1, 1].set_xlabel('RMSE')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].set_title('RMSE Distribution')

    plt.tight_layout()
    plt.savefig('seed_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()