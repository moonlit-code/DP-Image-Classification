# Differentially Private Image Classification (MNIST)
PyTorch + Opacus implementation of DP-SGD for MNIST classification.
Explores the **privacy-utility tradeoff** between privacy budget (ε, δ) and model accuracy.

## Project Overview
- Dataset: MNIST handwritten digits
- Model: Simple CNN (CPSC 340 style)
- Privacy Framework: DP-SGD via Opacus
- Goal: Measure how differential privacy impacts classification accuracy.

## Differential Privacy Basics
Differential Privacy (DP) protects individual training data by adding controlled noise to model gradients during training.
- **(ε, δ)**: Privacy budget pair
  - Smaller ε = stronger privacy protection
  - Larger ε = weaker privacy, higher model accuracy
- **DP-SGD**: Standard private training method combining gradient clipping + Gaussian noise injection.

## Environment Setup
```bash
python -m venv venv
# Activate virtual environment first
pip install torch torchvision opacus matplotlib numpy