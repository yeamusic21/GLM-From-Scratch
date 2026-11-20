# Insurance GLMs from Scratch

## About

A project dedicated to building Generalized Linear Models (GLMs) from scratch using NumPy, with a focus on understanding the underlying mathematics and mechanics of statistical modeling.

Rather than relying on high-level machine learning frameworks, this project implements the core components of GLMs directly. This provides full visibility into the model training process, including loss calculations, gradients, parameter updates, and model evaluation.

The initial focus is on insurance frequency and severity modeling using the [freMTPL2 dataset](https://www.kaggle.com/datasets/floser/french-motor-claims-datasets-fremtpl2freq).

### Learning Objectives

- Understand the mathematical foundations of Generalized Linear Models.
- Implement model training and gradient descent from scratch.
- Understand how link functions, offsets, weights, and exposure interact.
- Evaluate models using deviance and dispersion statistics.
- Explore overdispersion and its implications for insurance modeling.
- Develop an understanding of Poisson, Gamma, and Tweedie distributions.

## Environment

### Python Version

Built using Python 3.12.3.

### Create Environment (Ubuntu)

```bash
python3 -m venv .venv
```

### Activate Environment

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```