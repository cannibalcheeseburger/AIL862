
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
from scipy import ndimage
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

import sys

if len(sys.argv) > 1:
    folder_path = sys.argv[1]
else:
    folder_path = "images"


save_folder = 'segmentation_masks'
os.makedirs(save_folder, exist_ok=True)



if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model = model.to(device)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize((518, 518)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



def get_segmentation(image_path, num_segments=12):
    image_orig = Image.open(image_path).convert("RGB")
    image_np = np.array(image_orig)
    image_np = cv2.medianBlur(image_np, 5) 
    image = Image.fromarray(image_np)

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.forward_features(input_tensor)

    patch_tokens = features['x_norm_patchtokens'].squeeze(0).cpu()
    feature_map = patch_tokens.reshape(37, 37, -1)  

    reshaped_features = feature_map.reshape(-1, feature_map.shape[2])

    pca = PCA(n_components=0.99)  
    reduced_features = pca.fit_transform(reshaped_features)

    kmeans = KMeans(n_clusters=num_segments, random_state=0)
    segmentation_map = kmeans.fit_predict(reduced_features).reshape(37, 37)    
    segmentation_map = cv2.resize(segmentation_map, (512,512), interpolation=cv2.INTER_NEAREST)
    segmentation_map_smooth = ndimage.median_filter(segmentation_map, size=15)

    return image_orig,image, segmentation_map, segmentation_map_smooth



for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        image_path = os.path.join(folder_path, filename)
        image_orig, image, segmentation_map, segmentation_map_smooth = get_segmentation(image_path, 8)

        plt.imsave(os.path.join(save_folder, f"{filename.split('.')[0]}_segmentation_map.png"), segmentation_map, cmap='viridis')
        plt.imsave(os.path.join(save_folder, f"{filename.split('.')[0]}_segmentation_map_smooth.png"), segmentation_map_smooth, cmap='viridis')

        plt.figure(figsize=(20, 5))
        plt.suptitle(f"Results for {filename}")

        plt.subplot(1, 4, 1)
        plt.imshow(image_orig)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(image)
        plt.title("Median Blur")
        plt.axis('off')

        plt.subplot(1, 4, 3)
        plt.imshow(segmentation_map, cmap='viridis')
        plt.title("Semantic Segmentation")
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(segmentation_map_smooth, cmap='viridis')
        plt.title("Smoothed Segmentation")
        plt.axis('off')

        plt.show()

print("Segmentation maps saved in the 'segmentation masks' folder.")






