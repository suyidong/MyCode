"""
SISSO
Author: suyi dong
"""

# import TorchSisso model class along with other useful packages
from TorchSisso import SissoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl

# create dataframe composed of targets "y" and primary features "X"
traindata = pd.read_excel(r"D:\NO.1\PROCESS\二审修改2025.8.25（ddl9.14）\dataset -train.xlsx", sheet_name=0)
print(traindata)
testdata = pd.read_excel(r"D:\NO.1\PROCESS\二审修改2025.8.25（ddl9.14）\dataset - test.xlsx", sheet_name=0)
print(testdata)

#define unary and binary operators of interest
operators = ['+','-','*','/','||']

# create SISSO model object with relevant user-defined inputs
sm = SissoModel(traindata, operators, None, 2,4, 10)


# Run the SISSO algorithm to get the interpretable model with the highest accuracy
rmse, equation, r2,_ = sm.fit()
p,_ = sm.evaluate(equation,testdata)
print(p)
r_2 = r2_score(testdata.Target,p)
print('R2:', r_2 )



# Set up the plot style
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

# Evaluate equation for train and test data
def evaluate_equation(df, equation):
    return eval(equation, globals(), {**globals(), **df.to_dict('series')})

p_train = evaluate_equation(traindata, equation)
p_test = evaluate_equation(testdata, equation)

# Calculate metrics
r2_train = r2_score(traindata.Target, p_train)
r2_test = r2_score(testdata.Target, p_test)
rmse_train = np.sqrt(mean_squared_error(traindata.Target, p_train))
rmse_test = np.sqrt(mean_squared_error(testdata.Target, p_test))

# Create the parity plot
fig, ax = plt.subplots()

# Plot data points (swap x and y to match your requirement)
ax.scatter(p_train, traindata.Target, c='red', alpha=0.7, s=120, label=f'Train ($R^2$ = {r2_train:.3f})')
ax.scatter(p_test, testdata.Target, c='blue', alpha=0.7, s=120, label=f'Test ($R^2$ = {r2_test:.3f})')

# Set axis labels and customize ticks
ax.set_xlabel('Feathers', fontweight=1000, fontsize=28, fontname='Times New Roman')
ax.set_ylabel('Target', fontweight=1000, fontsize=28, fontname='Times New Roman')

# Customize tick labels
ax.tick_params(axis='both', which='major', labelsize=23)
for label in ax.get_xticklabels():
    label.set_fontweight('bold')
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Customize spines (plot borders)
for spine in ax.spines.values():
    spine.set_linewidth(3)
    spine.set_color('black')

# Plot the parity line (y = x)
min_val = min(min(traindata.Target), min(p_train), min(testdata.Target), min(p_test))
max_val = max(max(traindata.Target), max(p_train), max(testdata.Target), max(p_test))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.75, zorder=0, linewidth=4)

# Improve legend
ax.legend(loc='upper left', frameon=True, fancybox=True, framealpha=0.95, shadow=True, fontsize=24)

# Add minor ticks
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Ensure equal aspect ratio
ax.set_aspect('equal', 'box')

# Adjust layout and show
plt.tight_layout()
plt.show()