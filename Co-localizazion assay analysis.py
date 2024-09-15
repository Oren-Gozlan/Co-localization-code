import os
import re
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
import random



# This function gets paths of two channels from the same image, i this case red and blue, feel free to change. 
# It creates a mask based on a threshold you choose for each channel, change threshold as need.
def process_images(red_image_path, blue_image_path, red_percentage, blue_percentage):
    red_image = imageio.imread(red_image_path)
    blue_image = imageio.imread(blue_image_path)
    
    if red_image.shape[:2] != blue_image.shape[:2]:
        raise ValueError("The red and blue images must have the same dimensions.")

    red_threshold = np.percentile(red_image, red_percentage)
    blue_threshold = np.percentile(blue_image, blue_percentage)
    
    red_mask = red_image > red_threshold
    blue_mask = blue_image > blue_threshold    
    blue_red_mask = (blue_image > blue_threshold) & red_mask

    return red_mask, blue_mask, blue_red_mask

# Calculate the ratio between one mask and the combination of the two masks, feel fre to change which mask is divided by which as needed. 
def calculate_pixel_ratio(red_mask, blue_red_mask):
    total_red_pixels = np.sum(red_mask)
    blue_and_red_pixels = np.sum(blue_red_mask)
    ratio = blue_and_red_pixels / total_red_pixels
    return ratio

# Insert your main folder path and desired thresholds for each color:
main_folder = "Insert you main file location"
red_percentage = 95
blue_percentage = 85

# Define folder_names based on the folders in the main directory
folder_names = sorted([folder_name for folder_name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder_name))])

folder_means = []
folder_errors = []
all_ratios = []
folder_ratios = {}

# Iterate over folders in the specified main folder
for folder_name in folder_names:
    folder_path = os.path.join(main_folder, folder_name)
    
    if os.path.isdir(folder_path):
        subfolder_ratios = []
        
        # Iterate over subfolders in the folder
        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)

            if os.path.isdir(subfolder_path):
                ratios = []
                
                # Iterate over image pairs in the subfolder (Change "C1-" to your relevant name)
                # This is how the code knows to pair the channels that come from the same image.
                red_files = [f for f in os.listdir(subfolder_path) if f.startswith("C1-")]
                for red_file in red_files:
                    blue_file = re.sub(r'^C1-', 'C2-', red_file)
                    red_image_path = os.path.join(subfolder_path, red_file)
                    blue_image_path = os.path.join(subfolder_path, blue_file)

                    if os.path.exists(blue_image_path):
                        red_mask, blue_mask, blue_red_mask = process_images(red_image_path, blue_image_path, red_percentage, blue_percentage)
                        ratio = calculate_pixel_ratio(red_mask, blue_red_mask)
                        ratios.append(ratio)
                        all_ratios.append((folder_name, subfolder_name, red_file, ratio))

                subfolder_ratios.extend(ratios)
                folder_ratios[folder_name] = subfolder_ratios

        # Check if there are subfolders with image pairs before calculating the mean
        if subfolder_ratios:
            # Calculate the mean and mean error for the folder
            folder_mean = np.mean(subfolder_ratios)
            folder_means.append(folder_mean)

            folder_error = np.std(subfolder_ratios) / np.sqrt(len(subfolder_ratios))
            folder_errors.append(folder_error)

# Calculate the width of each bar dynamically based on the number of folders
bar_width = 0.8 / len(folder_names)  # Adjust the denominator as needed
bar_offsets = [-0.1,0.5] # Adjust the offset as needed


# Display the results in a bar plot with error bars and individual data points
fig, ax = plt.subplots(figsize=(5, 5))

bars = ax.bar(bar_offsets, folder_means, width=bar_width, yerr=folder_errors, color='none', edgecolor='black', linewidth = 2, capsize=5)

# Create a random pattern for the data points to spread around the error bar (change as needed)
dots_before = list(range(15,20))
dots_after = list(range(23,27))
dots_final = dots_before + dots_after

# Add data points for individual measurements inside the bars
dot_size = 4
for i, bar in enumerate(bars):
    for measurement in folder_ratios[folder_names[i]]:
        ax.plot(bar.get_x() + bar.get_width() - (random.choice(dots_final)/100),  
                measurement,
                marker='o',
                markersize=dot_size,
                color='black',
                markeredgecolor="none",
                linestyle='None')

# Name Y and X axis 
ax.set_ylabel('Y label', fontsize=15)
ax.set_xticks(bar_offsets)
ax.set_xticklabels(folder_names, rotation=20, ha='right', fontsize=15)

# Save the figure, change name as needed
plt.tight_layout()
plt.savefig("Change name as needed")
plt.show()

# Create a DataFrame from the list of ratios
df = pd.DataFrame(all_ratios, columns=['Main Folder', 'Subfolder', 'Image File', 'Pixel Ratio'])


# Export DataFrame to Excel
excel_filename = 'Change neame as needed.xlsx'  
df.to_excel(excel_filename, index=False)
print(f"\nData has been saved to {excel_filename}")

# Calculate Mann-Whitney U test p-values between main folders
main_folder_names = [folder_name for folder_name in os.listdir(main_folder) if os.path.isdir(os.path.join(main_folder, folder_name))]

# Extract all ratios for each main folder
folder_ratios = {folder_name: [] for folder_name in main_folder_names}
for row in all_ratios:
    folder_name = row[0]
    ratio = row[3]
    folder_ratios[folder_name].append(ratio)

# Calculate p-values between main folders
p_values = []
for i in range(0, len(main_folder_names), 2):
    folder1 = folder_ratios[main_folder_names[i]]
    folder2 = folder_ratios[main_folder_names[i + 1]]


    _, p_value = mannwhitneyu(folder1, folder2)
    p_values.append(p_value)

# Display p-values
print("\nP-values between main folders:")
for i, p_value in enumerate(p_values):
    print(f"{main_folder_names[i * 2]} vs {main_folder_names[i * 2 + 1]}: {p_value}")
