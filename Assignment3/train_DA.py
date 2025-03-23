import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
from torch.utils.data import Dataset
from PIL import Image
import os

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class UnlabeledDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(('.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image  
    

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

target_transform = transforms.Lambda(lambda y: torch.tensor(y, dtype=torch.long))

synthetic_dataset = ImageFolder(root="data/synthetic/cifar10", 
                                transform=transform_test,
                                target_transform=target_transform)
unlabeled_dataset = UnlabeledDataset("data/real/unlabelled", transform=transform_test)
test_dataset = ImageFolder(root="data/real/animal_data", transform=transform_test)

batch_size = 32
synthetic_loader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class FeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super(FeatureExtractor, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)

        for param in mobilenet.parameters():
            param.requires_grad = False
        
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    


num_classes = 3
model = FeatureExtractor(num_classes=num_classes).to(device)
model.eval()

def generate_pseudo_labels(model, dataloader, threshold=0.7):
    model.eval()
    pseudo_data = []

    with torch.no_grad():
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, pseudo_labels = torch.max(probabilities, dim=1)

            for i in range(len(images)):
                if confidence[i] > threshold:
                    pseudo_data.append((images[i].cpu(), pseudo_labels[i].cpu()))   
    
    return pseudo_data

class PseudoLabeledDataset(Dataset):
    def __init__(self, pseudo_data, transform=None):
        self.pseudo_data = pseudo_data
        self.transform = transform

    def __len__(self):
        return len(self.pseudo_data)

    def __getitem__(self, idx):
        image, label = self.pseudo_data[idx]
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
    

pseudo_data = generate_pseudo_labels(model, unlabeled_loader, threshold=0.51)
pseudo_dataset = PseudoLabeledDataset(pseudo_data, )  
combined_dataset = ConcatDataset([synthetic_dataset, pseudo_dataset])
combined_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)
num_epochs = 10

print("\nRetraining model with domain adaptation...")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in combined_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(combined_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

model.eval()
correct = 0
total = 0
test_loss = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_loss += loss.item()

test_accuracy = 100 * correct / total
test_loss /= len(test_loader)

print(f"\nFinal Test Accuracy: {test_accuracy:.2f}% | Test Loss: {test_loss:.4f}")

torch.save(model.state_dict(), "domain_adapted_model.pth")
print("\nImproved model saved as 'domain_adapted_model.pth'.")