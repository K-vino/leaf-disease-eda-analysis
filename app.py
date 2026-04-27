import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
from PIL import Image
from utils import load_data, process_subset

# Page Configuration
st.set_page_config(page_title="Leaf Disease EDA Dashboard", layout="wide", page_icon="🍃")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_value=True)

# --- Sidebar ---
st.sidebar.title("⚙️ Controls")
dataset_path = st.sidebar.text_input("Dataset Directory Path", value="./image data")
sample_limit = st.sidebar.slider("Sample Size for Analysis", 100, 2000, 500)

# Load Data
with st.spinner("Loading dataset structure..."):
    df = load_data(dataset_path)

if df.empty:
    st.error(f"Could not find images at: {dataset_path}. Please check the path.")
    st.stop()

selected_split = st.sidebar.selectbox("Select Dataset Split", df['split'].unique())
filtered_df = df[df['split'] == selected_split]

# --- Main App ---
st.title("🍃 Leaf Disease Exploratory Data Analysis")
st.markdown("---")

# A. Dataset Overview
st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Images", len(df))
with col2:
    st.metric(f"{selected_split.capitalize()} Set", len(filtered_df))
with col3:
    st.metric("Total Classes", df['label'].nunique())

st.subheader("Class Names")
st.write(", ".join(sorted(df['label'].unique())))

# B. Class Distribution
st.header("📈 Class Distribution")
fig_dist, ax_dist = plt.subplots(figsize=(12, 6))
sns.countplot(data=filtered_df, x='label', order=filtered_df['label'].value_counts().index, palette='viridis', ax=ax_dist)
plt.xticks(rotation=45)
st.pyplot(fig_dist)

# C. Sample Image Visualization
st.header("🖼️ Sample Visualizations")
selected_class = st.selectbox("Select Class to View Samples", sorted(filtered_df['label'].unique()))
num_samples = st.slider("Number of samples", 4, 12, 4)

class_samples = filtered_df[filtered_df['label'] == selected_class].sample(min(num_samples, len(filtered_df[filtered_df['label'] == selected_class])))
cols = st.columns(4)
for i, (_, row) in enumerate(class_samples.iterrows()):
    with cols[i % 4]:
        img = Image.open(row['image_path'])
        st.image(img, caption=f"Sample {i+1}", use_container_width=True)

# --- Perform Deep Analysis on Subset ---
st.markdown("---")
st.header("🔬 Technical Image Analysis")
st.info(f"Analyzing a random sample of {sample_limit} images from the {selected_split} set for performance.")

analysis_subset = filtered_df.sample(min(sample_limit, len(filtered_df)))
with st.spinner("Calculating image statistics..."):
    meta_df = process_subset(analysis_subset)

# D. Image Size Analysis
st.subheader("📏 Size Consistency")
unique_sizes = meta_df[['width', 'height']].drop_duplicates()
if len(unique_sizes) == 1:
    st.success(f"Perfect Consistency! All images are {unique_sizes.iloc[0]['width']}x{unique_sizes.iloc[0]['height']}")
else:
    st.warning("Variable sizes detected:")
    st.dataframe(unique_sizes)

# E. Color Analysis
st.subheader("🎨 Color Channel Distribution")
fig_color, ax_color = plt.subplots(figsize=(10, 5))
sns.kdeplot(meta_df['mean_r'], color='red', label='Red', ax=ax_color)
sns.kdeplot(meta_df['mean_g'], color='green', label='Green', ax=ax_color)
sns.kdeplot(meta_df['mean_b'], color='blue', label='Blue', ax=ax_color)
plt.legend()
st.pyplot(fig_color)

# F. Brightness Analysis
st.subheader("💡 Brightness Distribution")
fig_bright, ax_bright = plt.subplots(figsize=(10, 4))
sns.histplot(meta_df['brightness'], kde=True, color='gray', ax=ax_bright)
st.pyplot(fig_bright)

# G. Healthy vs Diseased Comparison
st.header("⚖️ Healthy vs Diseased")
comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    fig_comp, ax_comp = plt.subplots()
    df['condition'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], ax=ax_comp)
    plt.ylabel('')
    st.pyplot(fig_comp)

with comp_col2:
    fig_box, ax_box = plt.subplots()
    sns.boxplot(data=meta_df, x='condition', y='brightness', palette='Set2', ax=ax_box)
    plt.title("Brightness: Healthy vs Diseased")
    st.pyplot(fig_box)

# H. Insights & Recommendations
st.header("💡 Key Observations & Insights")
st.markdown(f"""
- **Dataset Balance**: The dataset has {df['label'].nunique()} classes. 
- **Quality**: { '✅ All images are standardized.' if len(unique_sizes) == 1 else '⚠️ Data requires resizing preprocessing.' }
- **Color Profile**: The Green channel peaks at {meta_df['mean_g'].mean():.2f}, indicating healthy chlorophyll presence.
- **Lighting**: Brightness spread is {meta_df['brightness'].std():.2f}. A high spread suggests you should use **Brightness Augmentation** during training.
- **Next Steps**: Based on this EDA, I recommend using a **Transfer Learning** approach with a model like **EfficientNet-B0** to handle the subtle texture differences in diseases.
""")

st.sidebar.markdown("---")
st.sidebar.write("Created by Vino K")
