Here's a detailed explanation of the **Lecture 26 slides** on advanced conditional generation techniques and efficient model adaptation:

---

## **1. Conditional Generation Methods**

### **a. Classifier Guidance**
A technique to steer diffusion models using gradients from a classifier during sampling.

#### **Algorithm Steps:**
1. Start with noise: $$ x_T \sim \mathcal{N}(0, \mathbf{I}) $$
2. For each timestep $$ t $$ from $$ T $$ to 1:
   - Compute denoising mean $$ \mu_\theta(x_t) $$ and variance $$ \Sigma_\theta(x_t) $$
   - Adjust the mean using classifier gradients:
     $$
     x_{t-1} \sim \mathcal{N}\left(\mu + s\Sigma \nabla_{x_t} \log p_\phi(y|x_t), \Sigma\right)
     $$
   - $$ s $$: Gradient scale (higher values enforce stronger class conditioning)

#### **Code Implementation:**
- Uses PyTorch autograd to compute gradients of classifier log-probs w.r.t. $$ x_t $$
- Modifies the diffusion sampling loop to incorporate these gradients

---

### **b. CLIP Guidance**
Similar to classifier guidance but uses CLIP's vision-language model:
- Perturbs the reverse process using gradients of the **image-text similarity score**
- Enables text-guided generation (e.g., "a photo of a corgi")

---

### **c. Classifier-Free Guidance**
Eliminates the need for a separate classifier by training a single model to handle both conditional and unconditional denoising.

#### **Key Equation:**
$$
\tilde{\epsilon}_\theta(z_\lambda, c) = (1+w)\epsilon_\theta(z_\lambda, c) - w\epsilon_\theta(z_\lambda)
$$
- $$ w $$: Guidance scale (controls strength of conditioning)
- Balances conditional ($$ \epsilon_\theta(z_\lambda, c) $$) and unconditional ($$ \epsilon_\theta(z_\lambda) $$) outputs

---

## **2. Latent Diffusion**
Operates in a compressed latent space rather than pixel space for efficiency.

#### **Workflow:**
1. **Encoder:** Compresses images to latent representations (e.g., using VAE)
2. **Diffusion:** Applies U-Net denoising in latent space
3. **Decoder:** Reconstructs high-resolution images from denoised latents

#### **Advantages:**
- Reduces computational costs by ~4x compared to pixel-space diffusion
- Uses cross-attention for conditioning (e.g., text prompts)

---

## **3. Visual-Prompt Tuning (VPT)**
A parameter-efficient method to adapt vision transformers (ViTs) to new tasks without full fine-tuning.

### **Key Idea:**
- **Freeze** the pre-trained ViT backbone
- **Prepend learnable prompt tokens** to the input sequence
- **Train only prompts + linear head** for downstream tasks

#### **VPT Variants:**
| Type         | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| **VPT-Shallow** | Adds prompts only at the input layer (first Transformer block)              |
| **VPT-Deep**    | Inserts prompts at **every Transformer block**, enabling deeper adaptation |

### **Code Implementation (VPT-Shallow):**
```python
class VPTWrapper(nn.Module):
    def __init__(self, vit_model, prompt_len, num_classes):
        super().__init__()
        self.vit = vit_model  # Frozen backbone
        self.prompt = nn.Parameter(torch.randn(1, prompt_len, embed_dim))
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.vit.patch_embed(x)  # Extract patches
        prompt_tokens = self.prompt.expand(x.size(0), -1, -1)
        x = torch.cat([self.vit.cls_token, prompt_tokens, x], dim=1)
        # Pass through frozen ViT blocks
        x = self.vit.blocks(x)  
        return self.classifier(x[:, 0])  # Classify using [CLS] token
```

### **Advantages Over Full Fine-Tuning:**
- **Storage Efficiency:** Only store prompts (0.1% of total parameters)
- **Prevents Catastrophic Forgetting:** Original weights remain frozen
- **Performance:** Outperforms linear probing and matches full fine-tuning on many tasks

---

## **4. Key Observations**
1. **Guidance Tradeoffs:**
   - Classifier guidance requires training separate classifiers
   - Classifier-free guidance simplifies workflow but needs careful weight tuning

2. **VPT Performance:**
   - VPT-Deep achieves **72.3% accuracy** vs 68.1% for full fine-tuning on FGVC-Aircraft
   - Works consistently across model sizes (ViT-Base/Large/Huge)

3. **Efficiency Gains:**
   - Latent diffusion reduces training costs by 75%
   - VPT reduces storage needs by 100-1000x for multi-task deployment

---

## **Practical Applications**
- **Text-to-Image Generation:** Stable Diffusion (latent diffusion + classifier-free guidance)
- **Medical Imaging:** Adapt pre-trained models to new modalities with VPT
- **Multi-Task Systems:** Deploy multiple prompt sets for different tasks on a single backbone

Let me know if you'd like clarification on specific diagrams or equations!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50850777/255dd8e0-597c-4199-a01f-a80beffebd78/lecture26.pdf

---
Answer from Perplexity: pplx.ai/share