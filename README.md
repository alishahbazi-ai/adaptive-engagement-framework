# The Adaptive Engagement Framework

This repository contains the data and source code associated with the study:

**The adaptive engagement framework: enhancing banking customer experience through AI-powered invisible marketing**

Published in *Scientific Reports* (2026).

## Repository Contents

The repository currently includes the following materials:

### Dataset

`BankDataSet_Processed.xlsx`

Processed banking customer dataset used for the computational experiments.

### Feature-to-Image Conversion

`Feature to image conversion.py`

Python script for converting the processed tabular data into image-based representations for deep learning classification.

### CNN Classification Pipeline

`Cnn classification pipeline.py`

Python implementation of the CNN classification pipeline used for model training and classification experiments.

### Requirements

`requirements.txt`

List of Python packages and dependencies required to run the provided code.

## Workflow

The main computational workflow is:

```text
BankDataSet_Processed.xlsx
            ↓
Feature to Image Conversion
            ↓
Image-Based Data
            ↓
CNN Classification Pipeline
            ↓
Model Training and Evaluation
```

## Installation

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Usage

Run the feature-to-image conversion script first:

```bash
python "Feature to image conversion.py"
```

Then run the CNN classification pipeline:

```bash
python "Cnn classification pipeline.py"
```

## Research Article

**The adaptive engagement framework: enhancing banking customer experience through AI-powered invisible marketing**

*Scientific Reports*, 2026.

## Citation

If you use the materials provided in this repository, please cite the associated research article.

## License

The materials in this repository are provided for research and academic purposes.
