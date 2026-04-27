import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import streamlit as st

@st.cache_data
def load_data(dataset_path):
    """Recursively find all images in the dataset path."""
    data_list = []
    if not os.path.exists(dataset_path):
        return pd.DataFrame()

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    
    for split in ['train', 'test', 'validation']:\
        split_path = os.path.join(dataset_path, split)
        if not os.path.exists(split_path):
            continue
            
        # Use os.walk to handle any level of nesting
        for root, dirs, files in os.walk(split_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    file_path = os.path.join(root, file)
                    
                    # Extract label information
                    # We'll use the folder name containing the image as the label
                    label = os.path.basename(root)
                    
                    # Also keep track of the parent folder (Crop)
                    parent_folder = os.path.basename(os.path.dirname(root))
                    
                    # Determine condition
                    condition = 'Healthy' if 'healthy' in label.lower() or 'healthy' in parent_folder.lower() else 'Diseased'
                    
                    data_list.append({
                        'image_path': file_path,
                        'label': label,
                        'crop': parent_folder if parent_folder != split else label,
                        'split': split,
                        'condition': condition
                    })
                    
    return pd.DataFrame(data_list)

def get_image_properties(image_path):
    """Extract size, color means, and brightness from an image."""
    try:
        # Check if it's a file, just in case
        if not os.path.isfile(image_path):
            return None
            
        img = Image.open(image_path)
        width, height = img.size
        img_array = np.array(img.convert('RGB')) # Ensure RGB
        
        mean_r = np.mean(img_array[:, :, 0])
        mean_g = np.mean(img_array[:, :, 1])
        mean_b = np.mean(img_array[:, :, 2])
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
