## Detailed Explanation of Lecture 16-17 Slides: BYOL, DINO, and MAE

These slides cover advanced self-supervised learning (SSL) methods and their applications in computer vision. Below, we break down the key concepts, architectures, and results.

---

### **1. BYOL (Bootstrap Your Own Latent)**

#### **Core Mechanism**
- **Two Networks:** 
  - **Online Network:** Updated via gradient descent.
  - **Target Network:** Updated via exponential moving average (EMA) of the online network.
- **Loss Function:** Maximizes similarity between differently augmented views of the same image:
  $$
  \mathcal{L} = -2 \cdot \left(\frac{\langle q_\theta(z_1), z'_1 \rangle}{\|q_\theta(z_1)\| \cdot \|z'_1\|} + \frac{\langle q_\theta(z_2), z'_2 \rangle}{\|q_\theta(z_2)\| \cdot \|z'_2\|}\right)
  $$
- **Avoids Negative Sampling:** Unlike contrastive methods (e.g., SimCLR), BYOL uses only positive pairs.

#### **Implementation Details**
- **Data Augmentation:** Includes resizing, color jitter, Gaussian blur, and grayscale conversion.
- **EMA Update:** The target network slowly tracks the online network:
  ```python
  class EMA:
      def update_average(self, old, new):
          return old * self.alpha + (1 - self.alpha) * new
  ```

#### **Performance Highlights**
- Outperforms supervised learning when fine-tuning with limited data:

| Method     | Top-1 Acc (1% Data) | Top-5 Acc (1% Data) |
|------------|---------------------|---------------------|
| Supervised | 25.4%               | 48.4%               |
| BYOL       | **53.2%**           | **78.4%**           |

- **Sensitivity:** Works best with:
  - **Projection dimension = 512**
  - **Batch size ≥ 256**

---

### **2. DINO (Self-Distillation with No Labels)**

#### **Key Innovations**
- **Vision Transformer (ViT) Backbone:** Uses self-attention to capture global image context.
- **Avoiding Collapse:** 
  - **Centering:** A running mean (C) of teacher outputs prevents degenerate solutions.
  - **Sharpening:** Applies a low temperature to teacher outputs to emphasize confident predictions.
- **Loss Function:** Cross-entropy between student and teacher outputs:
  ```python
  def H(t, s):
      t = softmax((t - C) / tpt, dim=1)  # Teacher: center + sharpen
      s = softmax(s / tps, dim=1)         # Student
      return - (t * log(s)).sum(dim=1).mean()
  ```

#### **Applications**
- **Semantic Segmentation:** Uses attention maps from ViT to identify object boundaries.
- **Copy Detection:** Achieves **85.5 mAP** on Copydays dataset with ViT-B/8.

#### **Performance**
- Outperforms BYOL and MoCo when using ViT-S:

| Method   | Architecture | Top-1 Acc (Linear) | k-NN Acc |
|----------|--------------|--------------------|----------|
| BYOL     | ViT-S        | 71.4%              | 66.6%    |
| DINO     | ViT-S        | **77.0%**          | **74.5%**|

---

### **3. MAE (Masked Autoencoder)**

#### **Architecture**
- **Encoder:** Processes only **25% of visible patches** (non-overlapping 16x16 patches).
- **Decoder:** Reconstructs masked patches from encoded visible patches and mask tokens.
- **Reconstruction Loss:** Mean squared error (MSE) between original and reconstructed pixels.

#### **Key Features**
- **High Masking Ratio:** 75% masking achieves best performance.
- **Non-Redundant Patches:** Non-overlapping patches force the model to learn global context.

#### **Results**
- Pre-trained MAE ViT-L outperforms supervised training:
  - **84.9% Top-1 Acc** vs. **82.5%** (scratch-trained ViT-L).

---

### **Critical Comparisons**

| Method | Strengths                          | Weaknesses                     |
|--------|------------------------------------|--------------------------------|
| BYOL   | No negative pairs; small-batch friendly | Sensitive to augmentation choices |
| DINO   | ViT compatibility; semantic features  | Requires careful hyperparameter tuning |
| MAE    | Scalable; strong reconstruction      | Computationally intensive decoder |

---

### **Practical Takeaways**
1. **BYOL** is ideal for limited-data scenarios.
2. **DINO** excels in tasks requiring semantic understanding (e.g., segmentation).
3. **MAE** is optimal for pre-training large ViTs on unlabeled data.

Let me know if you'd like further clarification on specific diagrams or code snippets!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50850777/a084ba62-26a4-4998-af0f-500b648dc734/lecture16And17.pdf

---
Answer from Perplexity: pplx.ai/share