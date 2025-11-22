# AIGP - Automated Genomic Phenotype Prediction Tool

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

## 📖 Project Introduction

AIGP (Automated Genomic Phenotype Prediction) is a machine learning-based genomic phenotype prediction tool designed for predicting phenotypic traits from genotypic data. The tool integrates multiple advanced machine learning algorithms, supports automated hyperparameter optimization, and provides detailed model interpretability analysis.

### ✨ Key Features

- **Multiple Format Support**: Supports various genotypic data formats (.txt, .ped, .vcf, .raw)
- **Rich Algorithms**: Integrates 13 machine learning algorithms (regression and classification)
- **Intelligent Optimization**: Built-in grid search and Sparrow Search Algorithm (SSA) for hyperparameter optimization
- **Dimensionality Reduction**: Supports PCA and PHATE dimensionality reduction methods
- **Model Interpretation**: Provides SHAP value analysis and feature importance evaluation
- **Parallel Computing**: Supports multi-core parallel processing for improved computational efficiency
- **GPU Acceleration**: Supports GPU acceleration for CatBoost, LightGBM, and XGBoost
- **Model Saving**: Supports saving and loading trained models

## 🚀 Quick Start

### Requirements

- Python 3.8+
- Recommended to use conda for environment management

### Installation Steps

1. **Clone the project**
```bash
git clone <repository-url>
cd AIGP
```

2. **Create conda environment**
```bash
conda create -n aigp python=3.8
conda activate aigp
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Train with default parameters
python main.py --genotype output.raw --phenotype phe_3549.txt

# Specify model and optimization method
python main.py --genotype output.raw --phenotype phe_3549.txt --model LightGBM --optimizer ssa

# Make predictions
python predict.py --model_path model.pkl --genotype new_data.raw --phenotype new_phenotype.txt
```

## 📁 Project Structure

```
AIGP/
├── aigp/                    # Core package
│   ├── __init__.py         # Package initialization
│   ├── cli.py              # Command-line interface
│   ├── data_loader.py      # Data loading module
│   ├── dim_reduction.py    # Dimensionality reduction module
│   ├── model_factory.py    # Model factory
│   ├── shap_analysis.py    # SHAP analysis module
│   ├── trainer.py          # Training module
│   └── utils.py            # Utility functions
├── data/                   # Example data
│   ├── test_x.txt         # Example genotypic data
│   └── test_y.txt         # Example phenotypic data
├── main.py                 # Main program entry
├── predict.py              # Prediction program
├── output.raw              # Example genotypic data (PLINK format)
├── phe_3549.txt            # Example phenotypic data
├── requirements.txt        # Dependency list
└── README.md              # Project documentation
```

## 🔧 Usage Instructions

> 📖 **Complete User Manual**: See [User Manual.md](User_Manual.md) for detailed usage guide and examples.  
> 📋 **Parameter Description**: See [Parameter Description.md](Parameter_Description.md) for detailed parameter descriptions.

### Quick Start

```bash
# Basic usage
python main.py --geno output.raw --phe phe_3549.txt --type regression --model LightGBM

# Cross-validation
python main.py --geno output.raw --phe phe_3549.txt --type regression --model LightGBM --cv 5

# SSA optimization
python main.py --geno output.raw --phe phe_3549.txt --type regression --model LightGBM --ssa

# Model prediction
python predict.py --model_path model.pkl --geno new_data.raw --phe new_phenotype.txt
```

## ✨ Key Features

- **Multiple Format Support**: Supports various genotypic data formats (.txt, .ped, .vcf, .raw)
- **Rich Algorithms**: Integrates 13 machine learning algorithms (regression and classification)
- **Intelligent Optimization**: Built-in grid search and Sparrow Search Algorithm (SSA) for hyperparameter optimization
- **Dimensionality Reduction**: Supports PCA and PHATE dimensionality reduction methods
- **Model Interpretation**: Provides SHAP value analysis and feature importance evaluation
- **Parallel Computing**: Supports multi-core parallel processing for improved computational efficiency
- **GPU Acceleration**: Supports GPU acceleration for CatBoost, LightGBM, and XGBoost
- **Model Saving**: Supports saving and loading trained models
- **Reproducibility**: Fixed random seed ensures reproducible results

## 📝 Changelog

### v1.1.0 (2024-12-19)
- ✅ **Fixed cross-validation random seed issue**: Ensures consistent results for each fold in 5-fold cross-validation, results are reproducible
- ✅ **Improved SHAP analysis**: Added error handling, performance optimization, and better interpreter selection
- ✅ **Cleaned project structure**: Removed unused test files, keeping the project clean
- ✅ **Improved documentation**: Created detailed user manual with all features and notes
- ✅ **Code optimization**: Improved error handling and code comments

### v1.0.0 (2024-01-XX)
- Initial version release
- Support for 13 machine learning algorithms
- Integrated SSA optimization algorithm
- Support for multiple data formats
- SHAP analysis functionality

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

- Project Link: [https://github.com/username/AIGP](https://github.com/username/AIGP)
- Issue Tracker: [https://github.com/username/AIGP/issues](https://github.com/username/AIGP/issues)

## 🙏 Acknowledgments

Thanks to all contributors and the open-source community for their support!

---

**Note**: This project is for academic research use only. Please ensure compliance with relevant data usage agreements and laws and regulations.
