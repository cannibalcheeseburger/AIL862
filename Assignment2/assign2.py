import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torch.utils.data import random_split
import torch
from torchvision.models import resnet18,vgg16
from torch import nn, optim
import random
import sys

    
#if torch.backends.mps.is_available():
#    device = torch.device('mps')
#    print("Using MPS (Metal Performance Shaders) device")
# Check for CUDA (NVIDIA GPU support)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA device")
# Fallback to CPU if neither MPS nor CUDA is available
else:
    device = torch.device('cpu')
    print("Using CPU device")

if len(sys.argv)==2:
    root_dir = sys.argv[1]
else:
    root_dir = "./Images/"

num_epochs = 1
batch_size = 32
class TiffDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for class_label in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_label)
            if os.path.isdir(class_path):  
                for image_name in os.listdir(class_path):
                    if image_name.endswith(".tif"):
                        image_path = os.path.join(class_path, image_name)
                        self.data.append([image_path, class_label])

        self.class_to_idx = {label: idx for idx, label in enumerate(sorted(set(label for _, label in self.data)))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, class_label = self.data[idx]
        image = Image.open(image_path).convert("RGB") 
        label = self.class_to_idx[class_label]

        if self.transform:
            image = self.transform(image)

        return image, label



def get_dataset(root_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),         
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
    ])

    dataset = TiffDataset(root_dir, transform=transform)
    return dataset

def add_noise(subset, original_dataset, noise_level):

    targets = original_dataset.data
    subset_indices = subset.indices  

    num_noisy = int(noise_level * len(subset))
    noisy_indices = random.sample(subset_indices, num_noisy)

    for idx in noisy_indices:
        current_label = targets[idx][1]
        possible_labels = list(original_dataset.class_to_idx.keys())
        possible_labels.remove(current_label)
        targets[idx][1] = random.choice(possible_labels)

    return subset

def train_model(model,train_loader,val_loader,criterion,optimizer,train_dataset,num_epochs):
    max_acc = 0
    min_loss = 100
    for epoch in range(num_epochs):
        model.train()  
        running_loss = 0.0
        
        for images, labels in train_loader:  
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_dataset):.4f}")

        model.eval()  
        validation_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  
            for images, labels in val_loader:  
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Validation Loss: {validation_loss / len(val_loader):.4f}, Accuracy: {acc:.2f}%")

        max_acc = max(max_acc,acc)
        min_loss = min(running_loss / len(train_dataset),min_loss)
    return max_acc,min_loss

def plot(noise_levels,values,title,model):
    plt.plot(noise_levels,values)
    plt.xlabel("Noise Level")
    plt.ylabel(title)
    plt.title(f"{title} vs. Noise Level")
    plt.legend()
    plt.savefig(f'{title}_{model}')
    plt.show()

resnet_stats = {}
noise_levels  = [0.01,0.05,0.1,0.2,0.4,0.6,0.9,1]
for noise_level in noise_levels:
    print('\n'+"="*8 +f'Finetuning ResNet18 With Noise Level {noise_level} '+"="*8+'\n')
    dataset = get_dataset(root_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset = add_noise(train_dataset, dataset, noise_level=noise_level)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.class_to_idx))  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model = model.to(device)
    acc, loss  = train_model(model,train_loader,val_loader,criterion,optimizer,train_dataset,num_epochs)
    resnet_stats[str(noise_level)] = [acc,loss]

accuracies = [acc for acc,loss in resnet_stats.values()]
losses = [loss for acc,loss in resnet_stats.values()]
plot(noise_levels,accuracies,'Accuracy','ResNet18')
plot(noise_levels,losses,'Loss','ResNet18')


vgg_stat = {}
noise_levels  = [0.01,0.05,0.1,0.2,0.4,0.6,0.9,1]
for noise_level in noise_levels:
    print('\n'+"="*8 +f'Finetuning VGG16 With Noise Level {noise_level} '+"="*8+'\n')
    dataset = get_dataset(root_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset = add_noise(train_dataset, dataset, noise_level=noise_level)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(dataset.class_to_idx))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model = model.to(device)
    acc, loss  = train_model(model,train_loader,val_loader,criterion,optimizer,train_dataset,num_epochs)
    vgg_stat[str(noise_level)] = [acc,loss]

vgg_accuracies = [acc for acc,loss in vgg_stat.values()]
vgg_losses = [loss for acc,loss in vgg_stat.values()]
plot(noise_levels,vgg_accuracies,'Accuracy','VGG16')
plot(noise_levels,vgg_losses,'Loss','VGG16')