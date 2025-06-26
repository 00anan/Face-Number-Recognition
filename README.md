# Face and Number Recognition with Naive Bayes & Perceptron

This project implements and compares two classic machine learning classifiers - **Naive Bayes** and a **Multi-Class Perceptron** - for identifying hand-drawn digits and human faces from text-based image data.

## Overview

The program loads digit and face image datasets represented in ASCII format, extracts features, and trains classifiers using different training sizes to evaluate performance in terms of accuracy and training time.

Key components:
- Custom **feature extraction** for digit (grid-based) and face (edge-density) data
- Binary **image preprocessing**
- Implementations of:
  - Naive Bayes Classifier
  - Multi-Class Perceptron Classifier
- Evaluation on both digit and face datasets with training size scaling

## Technologies

- **Python 3**
- **NumPy**
- **SQL** (for earlier data analysis, not used in final code)
- Terminal-based file I/O and ASCII-based datasets

## Dataset

The project expects a `data/` directory with two subdirectories:
- `digitdata/`: contains training, validation, and test sets for digits
- `facedata/`: contains training, validation, and test sets for faces

Each set includes ASCII-encoded image files and label files.

## How to Run
```bash
python final.py
```
Ensure your working directory contains the expected folder structure (e.g., `data/digitdata/trainingimages` and corresponding label files).

## Output
The program prints training time and classification accuracy for both classifiers across various training set sizes (10% to 100%) for both tasks.

#### Example output:

```bash 
...
Training size: 30%, Accuracy: 72.12%, Time: 0.001s
...
```

## Classifiers
#### Naive Bayes
- Binary features
- Laplace smoothing
- Assumes independence among features

#### Multi-Class Perceptron
- One-vs-all setup for classification
- Manual weight and bias updates
- Early stopping if convergence is achieved

## Notes
- This project was developed individually for an introductory artificial intelligence course.
- All classifier logic, feature extraction, and evaluation were implemented from scratch (no scikit-learn or external ML libraries).
- Intended primarily for academic use and learning purposes.

## Lessons Learned
- Preprocessing can significantly impact model performance
- Simple algorithms like Naive Bayes and Perceptron are still effective baselines
- Binary image classification is a great entry point for understanding ML pipelines
- Feel free to explore the code and use it as a reference for foundational ML techniques!