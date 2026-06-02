# Concrete Crack Detection

University computer vision project for binary classification of concrete surface images into **crack** and **non-crack** classes using deep learning.

The project will compare pretrained convolutional neural network models on two datasets:

- Concrete Crack Images for Classification
- SDNET2018

## Project Structure

- `data/` - local raw and processed datasets
- `notebooks/` - exploratory analysis notebooks
- `src/` - data loading, preprocessing, training, evaluation, and utilities
- `results/` - figures, metrics, trained model artifacts, and logs
- `report/` - LaTeX report files

## Setup

Use Python 3.11 or 3.12 for TensorFlow. Python 3.14 is too new for the TensorFlow package used in this project.

```bash
pip install -r requirements.txt
```

## Note

Dataset files are stored locally and are not committed to this repository because of their size.
