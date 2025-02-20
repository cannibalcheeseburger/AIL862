import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys

if len(sys.argv)>1:
    folder_path = sys.argv[1]
else: 
    folder_path = "./imagesAndReportFormat"


def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                images.append(img.copy())  
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
    return images

def get_embedding(image, model):
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        embedding = model(image).squeeze().numpy()  
    return embedding


def plot(images, pairs, similarity_scores):
    num_pairs = len(pairs)
    plt.figure(figsize=(7, num_pairs * 3))  
    
    for i, ((first, second), similarity) in enumerate(zip(pairs, similarity_scores)):
        plt.subplot(num_pairs, 3, i * 3 + 1)
        plt.imshow(images[first])
        plt.title(f"Image {first+1}", fontsize=12)
        plt.axis('off')
        
        plt.subplot(num_pairs, 3, i * 3 + 2)
        plt.text(0.5, 0.5, f"Similarity: {similarity:.4f}", 
                 fontsize=14, ha='center', va='center', 
                 bbox=dict(facecolor='white', edgecolor='black'))
        plt.axis('off')
        
        plt.subplot(num_pairs, 3, i * 3 + 3)
        plt.imshow(images[second])
        plt.title(f"Image {second+1}", fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


images = load_images(folder_path)
print(f"Loaded {len(images)} images.")

import torch
from torchvision import models, transforms

model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


softened_images = []
for image in images:
    image_np = np.array(image)
    image_np = cv2.medianBlur(image_np, 5) 
    softened_images.append(Image.fromarray(image_np))


embeddings = [get_embedding(img, model) for img in softened_images]

pairs = [(0,1),(0,3),(1,2),(0,4),(0,5)]
similarities =[]

print("="*8+" Results "+"="*8)
for first,second in pairs:
    similarity = cosine_similarity(embeddings[first],embeddings[second])
    similarities.append(similarity)
    print(f"Similarity between image {first+1} and image {second+1}:",similarity)
print("="*25)

plot(softened_images,pairs,similarities)