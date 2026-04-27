import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import streamlit as st

@st.cache_data
def load_data(dataset_path):
    """Crawl the dataset directory and return a DataFrame."""
    data_list = []
    if not os.path.exists(dataset_path):
        return pd.DataFrame()

    for split in ['train', 'test', 'validation']:
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    data_list.append({
                        'image_path': os.path.join(class_path, image_name),
                        'label': class_name,
                        'split': split,
                        'condition': 'Healthy' if 'healthy' in class_name.lower() else 'Diseased'
                    })
    return pd.DataFrame(data_list)

def get_image_properties(image_path):
    """Extract size, color means, and brightness from an image."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        img_array = np.array(img)
        
        # Handle grayscale or RGBA images
        if len(img_array.shape) == 3:
            mean_r = np.mean(img_array[:, :, 0])
            mean_g = np.mean(img_array[:, :, 1])
            mean_b = np.mean(img_array[:, :, 2])
        else:
            mean_r = mean_g = mean_b = np.mean(img_array)
            
        brightness = np.mean(img_array)
        return width, height, mean_r, mean_g, mean_b, brightness
    except Exception as e:
        return None

@st.cache_data
def process_subset(df_subset):
    """Process a list of images to get metadata."""
    results = []
    for _, row in df_subset.iterrows():
        props = get_image_properties(row['image_path'])
        if props:
            results.append({
                'label': row['label'],
                'condition': row['condition'],
                'width': props[0],
                'height': props[1],
                'mean_r': props[2],
                'mean_g': props[3],
                'mean_b': props[4],
                'brightness': props[5]
            })
    return pd.DataFrame(results)
