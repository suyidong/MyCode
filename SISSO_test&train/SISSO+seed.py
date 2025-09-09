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

traindata = pd.read_excel('way\to\your\path', sheet_name=0)
testdata = pd.read_excel('way\to\your\path', sheet_name=0)

operators = ['+', '-', '*', '/', '||']

results = []

SEED_RANGE = 20 


sm_original = SissoModel(traindata, operators, None, 2, 4, 10)
rmse_original, equation_original, r2_original, _ = sm_original.fit()
p_test_original, _ = sm_original.evaluate(equation_original, testdata)
r2_test_original = r2_score(testdata.Target, p_test_original)
rmse_original_test = np.sqrt(mean_squared_error(testdata.Target, p_test_original))

print(f"train_R²: {r2_original:.4f}")
print(f"test_R²: {r2_test_original:.4f}")
print(f"equation: {equation_original}")
results.append({
            'seed': -1,
            'r2_train': r2_original,
            'r2_test': r2_test_original,
            'rmse_train': rmse_original,
            'rmse_test': rmse_original_test,
            'equation': equation_original
        })

fulldata = pd.concat([traindata, testdata], ignore_index=True)

for seed in range(SEED_RANGE):
    print(f"\n {seed + 1}/{SEED_RANGE}")

    try:
        shuffled_data = fulldata.sample(frac=1, random_state=seed).reset_index(drop=True)

        split_idx = int(0.8 * len(shuffled_data))
        train_data = shuffled_data.iloc[:split_idx]
        test_data = shuffled_data.iloc[split_idx:]

        sm = SissoModel(train_data, operators, None, 2, 4)

        rmse, equation, r2, _ = sm.fit()
        print(f"equation: {equation}")

        p_test, _ = sm.evaluate(equation, test_data)

        r2_train = r2
        r2_test = r2_score(test_data.Target, p_test)
        rmse_train = rmse
        rmse_test = np.sqrt(mean_squared_error(test_data.Target, p_test))

        results.append({
            'seed': seed,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'rmse_train': rmse_train,
            'rmse_test': rmse_test,
            'equation': equation
        })

    except Exception as e:
        print(f"seed {seed} failed: {str(e)}")
        continue

if not results:
    print("All seed failed. Please check.")
    exit()

print("\nresults:")
print(f"success: {len(results)}/{SEED_RANGE}")
print(f"avg_train_R²: {np.mean([r['r2_train'] for r in results]):.4f} ± {np.std([r['r2_train'] for r in results]):.4f}")
print(f"avg_test_R²: {np.mean([r['r2_test'] for r in results]):.4f} ± {np.std([r['r2_test'] for r in results]):.4f}")
print(
    f"avg_train_RMSE: {np.mean([r['rmse_train'] for r in results]):.4f} ± {np.std([r['rmse_train'] for r in results]):.4f}")
print(
    f"avg_test_RMSE: {np.mean([r['rmse_test'] for r in results]):.4f} ± {np.std([r['rmse_test'] for r in results]):.4f}")

equations = [r['equation'] for r in results]
unique_equations = set(equations)
for i, eq in enumerate(unique_equations):
    print(f"equation {i + 1}: {eq}")
    count = equations.count(eq)
    print(f"times: {count} ({count / len(results) * 100:.1f}%)")

results_df = pd.DataFrame(results)
results_df = results_df[['seed', 'r2_train', 'r2_test', 'rmse_train', 'rmse_test', 'equation']]

stats_data = {
    'Metric': ['train_R²', 'test_R²', 'train_RMSE', 'test_RMSE'],
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

eq_freq_data = []
for eq in unique_equations:
    count = equations.count(eq)
    eq_freq_data.append({
        'Equation': eq,
        'Frequency': count,
        'Percentage': f"{count / len(results) * 100:.1f}%"
    })
eq_freq_df = pd.DataFrame(eq_freq_data)

with pd.ExcelWriter('seed_sensitivity_analysis_results.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='detail', index=False)
    stats_df.to_excel(writer, sheet_name='information', index=False)
    eq_freq_df.to_excel(writer, sheet_name='frequency', index=False)

print("\n 'seed_sensitivity_analysis_results.xlsx'")

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

if len(results) > 1:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot([r['seed'] for r in results], [r['r2_train'] for r in results], 'ro-', alpha=0.7, label='Train R²')
    axes[0, 0].plot([r['seed'] for r in results], [r['r2_test'] for r in results], 'bo-', alpha=0.7, label='Test R²')
    axes[0, 0].axhline(y=r2_original, color='g', linestyle='--', label='Original Train R²')
    axes[0, 0].axhline(y=r2_test_original, color='m', linestyle='--', label='Original Test R²')
    axes[0, 0].set_xlabel('Seed')
    axes[0, 0].set_ylabel('R²')
    axes[0, 0].legend()
    axes[0, 0].set_title('R² vs Seed')

    axes[0, 1].plot([r['seed'] for r in results], [r['rmse_train'] for r in results], 'ro-', alpha=0.7,
                    label='Train RMSE')
    axes[0, 1].plot([r['seed'] for r in results], [r['rmse_test'] for r in results], 'bo-', alpha=0.7,
                    label='Test RMSE')
    axes[0, 1].set_xlabel('Seed')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].legend()
    axes[0, 1].set_title('RMSE vs Seed')

    axes[1, 0].hist([r['r2_train'] for r in results], bins=20, alpha=0.7, label='Train R²', color='red')
    axes[1, 0].hist([r['r2_test'] for r in results], bins=20, alpha=0.7, label='Test R²', color='blue')
    axes[1, 0].axvline(x=r2_original, color='g', linestyle='--', label='Original Train R²')
    axes[1, 0].axvline(x=r2_test_original, color='m', linestyle='--', label='Original Test R²')
    axes[1, 0].set_xlabel('R²')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].set_title('R² Distribution')

    axes[1, 1].hist([r['rmse_train'] for r in results], bins=20, alpha=0.7, label='Train RMSE', color='red')
    axes[1, 1].hist([r['rmse_test'] for r in results], bins=20, alpha=0.7, label='Test RMSE', color='blue')
    axes[1, 1].set_xlabel('RMSE')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].set_title('RMSE Distribution')

    plt.tight_layout()
    plt.savefig('seed_sensitivity_analysis.png', dpi=300, bbox_inches='tight')

    plt.show()
