# The Adaptive Engagement Framework

Official research code, computational resources, and reproducibility materials for the study:

> **The adaptive engagement framework: enhancing banking customer experience through AI-powered invisible marketing**

Published in *Scientific Reports*.

## Overview

This repository contains the source code, documentation, computational workflows, and supporting resources associated with the study.

The study investigates an AI-based framework for customer engagement in digital banking and evaluates classical machine learning and deep learning approaches for predicting customer interaction propensity.

The complete computational pipeline includes data preprocessing, feature selection, class-imbalance handling, classical machine learning, feature-to-image transformation, convolutional neural networks, and the proposed RDAD-CNN architecture.

## Dataset

The study uses a customer-level dataset comprising **45,211 records**.

Due to privacy, confidentiality, and institutional data-access restrictions, the original customer-level dataset is not publicly distributed in this repository.

Researchers with authorized access to the dataset can reproduce the computational workflow by placing the permitted dataset in the designated:

```text
data/raw/
```

directory.

Dataset preparation requirements and expected file formats are described in:

```text
data/README.md
```

## Computational Workflow

The repository implements the following computational stages:

1. Data preprocessing
2. Class-imbalance handling
3. Mutual Information-based feature selection
4. Classical machine learning classification
5. Feature-to-image transformation
6. Deep learning classification
7. RDAD-CNN implementation
8. Cross-validation
9. Performance evaluation
10. Statistical analysis
11. Result visualization

## Machine Learning Models

### Classical Machine Learning

The classical machine learning experiments include:

* Random Forest
* Decision Tree
* Support Vector Machine
* Deep Neural Network

### Deep Learning

The deep learning experiments include:

* Plain CNN
* CNN-SE
* RDAD-CNN

## Repository Structure

```text
adaptive-engagement-framework/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── visualization/
│
├── notebooks/
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── metrics/
│
└── docs/
```

## Reproducibility

The repository is organized to support reproducible execution of the computational experiments reported in the study. The source code is separated into data processing, model development, training, evaluation, and visualization components.

Because the original customer-level dataset is subject to privacy and access restrictions, complete reproduction requires authorized access to the underlying dataset. The repository therefore provides the computational pipeline and supporting resources required to reproduce the analyses when the permitted data are available.

## Software Requirements

The required Python packages and their versions are specified in:

```text
requirements.txt
```

The recommended workflow is to create an isolated Python environment and install the required dependencies before executing the preprocessing, training, and evaluation scripts.

## Citation

If you use this repository or its computational resources in your research, please cite the associated publication:

> **The adaptive engagement framework: enhancing banking customer experience through AI-powered invisible marketing.** *Scientific Reports*.

The machine-readable citation information is also provided in:

```text
CITATION.cff
```

## License

The source code in this repository is distributed under the terms specified in the `LICENSE` file.

## Data Availability

The underlying customer-level dataset is not publicly available because of privacy, confidentiality, and institutional restrictions. Access to the data may be available to qualified researchers subject to the applicable institutional and data-governance requirements.

## Contact

For questions regarding the computational framework, reproducibility, or authorized use of the research resources, please refer to the contact information provided in the associated publication.
