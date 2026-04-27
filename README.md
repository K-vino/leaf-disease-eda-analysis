# 🍃 Leaf Disease Dataset: Exploratory Data Analysis (EDA)

![Project Header](https://img.shields.io/badge/Data%20Science-EDA-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📋 Project Overview
This project performs a comprehensive Exploratory Data Analysis on a massive dataset of over **53,000 images** of plant leaves across **13 different classes**. The goal is to understand the underlying patterns, data quality, and visual features that distinguish healthy leaves from diseased ones.

This analysis serves as a foundational step for building a robust Deep Learning model for automated crop disease detection.

## 🚀 Key Features
- **Data Structuring**: Converted raw image directories into a searchable Pandas DataFrame.
- **Distribution Analysis**: Visualized class balance and dataset splits (Train/Test/Val).
- **Visual Exploration**: Generated automated sample grids for each leaf category.
- **Image Metadata Analysis**: Extracted and analyzed image dimensions, color distributions (RGB), and brightness levels.
- **Comparative Study**: Performed a direct statistical and visual comparison between 'Healthy' and 'Diseased' samples.

## 📂 Dataset Structure
The dataset is organized into three main splits:
- `train/`: 13 classes of leaf images.
- `test/`: Unseen images for final evaluation.
- `validation/`: Images used for tuning.

**Classes include:** Cassava, Rice, Apple, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Squash, Strawberry, and Tomato.

## 📊 Key Insights
- **Image Consistency**: All images are standardized at 256x256 pixels, reducing the need for heavy preprocessing.
- **Color Signature**: Green channel intensity is a primary differentiator for leaf health.
- **Balance**: Identified specific classes with lower sample counts that require data augmentation strategies.

## 🛠️ Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/leaf-disease-eda-analysis.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   Open `notebooks/EDA.ipynb` in Jupyter Notebook or VS Code and run all cells.

## 🧪 Tools Used
- **Pandas & NumPy**: Data manipulation.
- **Matplotlib & Seaborn**: Static visualizations.
- **OpenCV & PIL**: Image processing.
- **Tqdm**: Progress tracking for large-scale analysis.

---
*Created by [Your Name] as part of a Data Science Portfolio project.*
