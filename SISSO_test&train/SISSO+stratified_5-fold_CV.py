"""
Stratified 5-fold CV with SISSO
Author: suyi dong
"""


import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold
from matplotlib.ticker import AutoMinorLocator
import torch
from TorchSisso import SissoModel 


DATA_PATH = 'way/to/your/path'
df_all = pd.read_excel(DATA_PATH, sheet_name=0)
print('Total samples:', len(df_all))
print(df_all.head())


feature_cols = [c for c in df_all.columns if c != 'Target']
X_all = df_all[feature_cols]
y_all = df_all['Target']



y_bin = pd.cut(y_all, bins=5, labels=False)


operators = ['+', '-', '*', '/', '||']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=224)

fold = 0
all_y_true, all_y_pred = [], []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
for train_idx, val_idx in skf.split(X_all, y_bin):
    fold += 1
    print(f'\n===== Fold {fold}/5 =====')


    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_val = df_all.iloc[val_idx].reset_index(drop=True)


    sm = SissoModel(df_train, operators, None, 2, 4)

    rmse_train, equation, r2_train, _ = sm.fit()
    print('SISSO equation:', equation)


    y_pred_val, _ = sm.evaluate(equation, df_val)
    r2_val = r2_score(df_val.Target, y_pred_val)
    rmse_val = np.sqrt(mean_squared_error(df_val.Target, y_pred_val))

    print(f'Fold {fold}  R²={r2_val:.4f}  RMSE={rmse_val:.4f}')

    
    all_y_true.extend(df_val.Target.values)
    all_y_pred.extend(y_pred_val)


r2_overall = r2_score(all_y_true, all_y_pred)
rmse_overall = np.sqrt(mean_squared_error(all_y_true, all_y_pred))
print('\n========== Overall 5-Fold CV ==========')
print(f'R²  = {r2_overall:.4f}')
print(f'RMSE= {rmse_overall:.4f}')
df = pd.DataFrame({
    'y_true': all_y_true,
    'y_pred': all_y_pred
})

df.to_excel('cv_results.xlsx', index=False)
print('cv_results.xlsx')

plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 20,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 14,
    'figure.dpi': 300,
    'figure.figsize': (10, 10),
    'axes.grid': True,
    'grid.alpha': 0.3
})

fig, ax = plt.subplots()
ax.scatter(all_y_pred, all_y_true,
           c='tab:blue', alpha=0.7, s=120,
           label=f'5-Fold CV (R² = {r2_overall:.3f})')


min_v, max_v = min(all_y_true), max(all_y_true)
ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2)

ax.set_xlabel('Predicted', fontweight=1000, fontsize=28, fontname='Times New Roman')
ax.set_ylabel('Observed', fontweight=1000, fontsize=28, fontname='Times New Roman')

ax.tick_params(axis='both', which='major', labelsize=23)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

ax.legend(loc='upper left', frameon=True, fancybox=True,
          framealpha=0.95, shadow=True, fontsize=24)
ax.set_aspect('equal', 'box')
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

plt.tight_layout()

plt.show()
