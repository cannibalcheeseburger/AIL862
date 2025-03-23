import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

batch_size = 32

from torch.utils.data import Dataset

augmentation_transforms = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = ImageFolder(root="data/synthetic/cifar10",transform=transform)
test_dataset = ImageFolder(root="data/real/animal_data", transform=transform_test)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class FeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super(FeatureExtractor, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        
        for param in mobilenet.parameters():
            param.requires_grad = False
        
        self.features = mobilenet.features
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)  
        )

    def forward(self, x):
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

num_classes = 3  
num_epochs = 10

total_samples = len(train_dataset)

n_val = int(0.2 * total_samples)
n_train = total_samples - n_val
train_data, val_data = random_split(train_dataset, [n_train, n_val])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

model = FeatureExtractor(num_classes=3).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
print('starting training')

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss_sum = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        train_loss_sum += loss.item()
    scheduler.step()
    train_accuracy = 100 * train_correct / train_total
    train_loss = train_loss_sum / len(train_loader)
    
    model.eval()
    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            val_loss_sum += loss.item()
    
    val_accuracy = 100 * val_correct / val_total
    val_loss = val_loss_sum / len(val_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

model.eval()
test_correct = 0
test_total = 0
test_loss_sum = 0.0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        test_loss_sum += loss.item()

test_accuracy = 100 * test_correct / test_total
test_loss = test_loss_sum / len(test_loader)
test_error = 100 - test_accuracy

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Error: {test_error:.2f}%')

