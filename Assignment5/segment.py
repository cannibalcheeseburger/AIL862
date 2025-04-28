
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm
from torchvision.transforms import functional as F



torch.manual_seed(420)


DATASET_ROOT = 'dataset'
D_DIR = os.path.join(DATASET_ROOT, 'd')

DOG_CLASS_INDEX = 12  

item_d_imgs = sorted([os.path.join(D_DIR, fname) for fname in os.listdir(D_DIR) if fname.lower().endswith('.jpg')])
item_d_masks = [img.replace('.jpg', '_mask.png') for img in item_d_imgs]


pseudo_label_dir = 'pseudo_labels'
ORIG_DIR = 'orig'  

likely_dog_imgs = [os.path.join(ORIG_DIR, fname)
                   for fname in os.listdir(ORIG_DIR)
                   if fname.lower().endswith('.jpg')]


class PseudoLabeledDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        mask = np.array(mask)
        mask = np.where(mask > 127, 1, 0).astype(np.uint8)
        mask = Image.fromarray(mask)
        inputs = self.processor(images=image, segmentation_maps=mask, return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs


processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
pseudo_label_paths = [os.path.join(pseudo_label_dir, os.path.basename(img).replace('.jpg', '_mask.png')) for img in likely_dog_imgs]
dataset = PseudoLabeledDataset(likely_dog_imgs, pseudo_label_paths, processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-ade-640-640",
    num_labels=2,  
    ignore_mismatched_sizes=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


model.train()
for epoch in range(10):  
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].long().to(device)
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())



model.save_pretrained(os.path.join(DATASET_ROOT, "finetuned_segformer_dog"))
processor.save_pretrained(os.path.join(DATASET_ROOT, "finetuned_segformer_dog"))


def compute_iou(pred_mask, true_mask, num_classes=2):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred_mask == cls)
        true_inds = (true_mask == cls)
        intersection = (pred_inds & true_inds).sum().item()
        union = (pred_inds | true_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # Ignore this class
        else:
            ious.append(intersection / union)
    return ious

def pixel_accuracy(pred_mask, true_mask):
    correct = (pred_mask == true_mask).sum().item()
    total = true_mask.numel()
    return correct / total

def evaluate(model, img_list, mask_list, processor, device):
    model.eval()
    iou_list = []
    pixel_acc_list = []
    image_metrics = []  

    with torch.no_grad():
        for img_path, mask_path in zip(img_list, mask_list):
            image = Image.open(img_path).convert("RGB")
            true_mask = Image.open(mask_path)
            true_mask = np.array(true_mask)
            true_mask = np.where(true_mask == DOG_CLASS_INDEX, 0, 1).astype(np.uint8)
            true_mask_tensor = torch.tensor(true_mask, dtype=torch.int64)
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device)
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits
            pred_mask = torch.argmax(logits.squeeze(), dim=0).cpu()

            if true_mask_tensor.shape != pred_mask.shape:
                true_mask_pil = Image.fromarray(true_mask_tensor.numpy().astype(np.uint8))
                true_mask_resized = F.resize(true_mask_pil, pred_mask.shape, interpolation=Image.NEAREST)
                true_mask_tensor = torch.tensor(np.array(true_mask_resized), dtype=torch.int64)

            ious = compute_iou(pred_mask, true_mask_tensor, num_classes=2)
            iou_list.append(ious)
            pixel_acc = pixel_accuracy(pred_mask, true_mask_tensor)
            pixel_acc_list.append(pixel_acc)

            image_metrics.append({
                "img_path": img_path,
                "iou": ious[1],  
                "pixel_acc": pixel_acc
            })

    iou_array = np.array(iou_list)
    mean_iou_per_class = np.nanmean(iou_array, axis=0)
    mean_iou = np.nanmean(mean_iou_per_class)
    mean_pixel_acc = np.mean(pixel_acc_list)
    return {
        "miou": mean_iou,
        "miou_per_class": mean_iou_per_class.tolist(),
        "pixel_acc": mean_pixel_acc,
        "image_metrics": image_metrics  
    }

metrics = evaluate(model, item_d_imgs, item_d_masks, processor, device)
print("mIoU:", metrics['miou'], "Pixel Accuracy:", metrics['pixel_acc'])

