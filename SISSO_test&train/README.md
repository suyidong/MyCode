# SISSO Model with Cross-Validation and Seed Sensitivity Analysis

This repository contains Python scripts for running the SISSO (Sure Independence Screening and Sparsifying Operator) algorithm with additional validation techniques including stratified 5-fold cross-validation and seed sensitivity analysis.

## Overview

The SISSO algorithm is a powerful method for feature selection and model construction in scientific domains where interpretable models are essential. These scripts offer the basic SISSO implementation using TorchSisso.
## Files Description

### 1. SISSO+seed.py
Performs seed sensitivity analysis on SISSO models by testing multiple random seeds for data shuffling and model training. This helps evaluate the stability of SISSO models against different data partitions.

**Key features:**
- Tests multiple random seeds (default: 20)
- Compares performance metrics across different data splits
- Generates comprehensive visualizations and Excel reports
- Analyzes equation consistency across different seeds

### 2. SISSO+stratified_5-fold_CV.py
Implements stratified 5-fold cross-validation for SISSO models. This approach ensures that each fold maintains the same distribution of the target variable, providing a more reliable performance estimate.

**Key features:**
- Stratified sampling based on target variable distribution
- 5-fold cross-validation with consistent evaluation metrics
- GPU support detection and utilization
- Comprehensive parity plots for visualization

### 3. sissotext.py
Basic implementation of the SISSO algorithm for training and testing on fixed datasets. This serves as the foundation for the more advanced validation techniques.

**Key features:**
- Standard SISSO model training and evaluation
- Customizable operator set
- Detailed parity plot visualization
- Performance metrics calculation

## Requirements

- Python 3.7+
- TorchSisso
- pandas
- numpy
- scikit-learn
- matplotlib
- sympy
- torch

Install required packages:
```bash
pip install pandas numpy scikit-learn matplotlib sympy torch
```

For TorchSisso, please follow installation instructions from the official repository.

## Usage

### Basic SISSO Model
```bash
python sissotext.py
```

### Seed Sensitivity Analysis
```bash
python SISSO+seed.py
```

### Stratified 5-Fold Cross-Validation
```bash
python SISSO+stratified_5-fold_CV.py
```

## Data Format

Input data should be in Excel format with the following structure:
- Features: Multiple columns with descriptive names
- Target: A column named "Target" containing the dependent variable

Example dataset structure:
| Feature1 | Feature2 | Feature3 | ... | Target |
|----------|----------|----------|-----|--------|
| 1.2      | 0.5      | 3.4      | ... | 5.6    |
| 2.1      | 1.2      | 4.1      | ... | 7.2    |

## Customization

### Operators
You can customize the set of operators used by SISSO by modifying the `operators` list:
```python
operators = ['+', '-', '*', '/', '||']  # Add or remove operators as needed
```

### Model Parameters
Adjust SISSO parameters in the model initialization:
```python
sm = SissoModel(traindata, operators, None, 2, 4, 10)
# Parameters: data, operators, constraints, n_dim, n_sisso, n_feature
```

## Output

The scripts generate:
1. Performance metrics (R², RMSE) for training and testing
2. Visualizations (parity plots, sensitivity analysis charts)
3. Excel files with detailed results
4. Equation frequency analysis (seed sensitivity script)

## Citation

If you use this code in your research, please cite the original SISSO paper and consider referencing this implementation.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For questions or issues, please open an issue in the GitHub repository.
