Here's a detailed explanation of the **Lecture 25 slides** on Diffusion Models (DDPMs):

---

## **1. Forward Diffusion Process**
The forward process gradually adds Gaussian noise to data over $$ T $$ timesteps.

### **Key Equations**
- **Single Step:**  
  $$
  q(x_t | x_{t-1}) = \mathcal{N}\left(x_t \mid \sqrt{1-\beta_t} x_{t-1}, \beta_t \mathbf{I}\right)
  $$
  - $$ \beta_t $$: Noise schedule (predefined variance).  
  - $$ \sqrt{1-\beta_t} $$: Shrinks previous state to preserve signal.  

- **Marginal Distribution (Closed Form):**  
  $$
  q(x_t | x_0) = \mathcal{N}\left(x_t \mid \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)\mathbf{I}\right)
  $$
  - $$ \alpha_t = 1 - \beta_t $$, $$ \bar{\alpha}_t = \prod_{s=1}^t \alpha_s $$.  
  - Allows direct sampling of $$ x_t $$ from $$ x_0 $$:  
    $$
    x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon \quad (\epsilon \sim \mathcal{N}(0, \mathbf{I}))
    $$

### **Code Implementation**
```python
# Precompute noise schedule parameters
T = 1000
alphas = torch.linspace(start=0.9999, end=0.98, steps=T, dtype=torch.float64).to(device)
alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t = α_1 * α_2 * ... * α_t
sqrt_alpha_bars_t = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars_t = torch.sqrt(1.0 - alpha_bars)

# Sample x_t from x_0
x_t = sqrt_alpha_bars_t[t] * x_0 + sqrt_one_minus_alpha_bars_t[t] * eps  # eps ~ N(0,I)
```

---

## **2. Reverse Diffusion (Decoder)**
Learns to reverse the noise-adding process using a neural network (U-Net).

### **Training Objective**
Minimize MSE between predicted noise $$ \epsilon_\theta $$ and true noise $$ \epsilon $$:  
$$
\nabla_\theta \left\| \epsilon - \epsilon_\theta\left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t \right) \right\|^2
$$

### **U-Net Architecture**
- **Input:** Noisy image $$ x_t $$ and timestep embedding $$ t $$.  
- **Timestep Embedding:**  
  - Convert $$ t $$ to sinusoidal embeddings.  
  - Pass through MLP and add to intermediate layers via `out = out + pos`.  
- **Residual Blocks:** Use skip connections and Swish (SiLU) activations.  

```python
def forward(self, x, t_emb):
    out = self.norm1(x)
    out = F.silu(out)  # Swish activation
    out = self.conv1(out)
    
    # Add timestep embedding
    pos = F.silu(t_emb)
    pos = self.linear_pos(pos)
    pos = pos[:, :, None, None]  # Reshape for broadcasting
    out = out + pos  # Feature modulation by time
    
    # Residual connection
    out = self.norm2(out)
    out = F.silu(out)
    out = self.conv2(out)
    return out + x  # Skip connection
```

---

## **3. Training Algorithm**
```
1. Sample clean data x_0 ~ q(x_0)
2. Sample timestep t ~ Uniform{1, ..., T}
3. Sample noise ε ~ N(0, I)
4. Compute x_t = sqrt(ᾱ_t) x_0 + sqrt(1-ᾱ_t) ε
5. Predict ε_θ(x_t, t) with U-Net
6. Update θ via MSE loss: ||ε - ε_θ||²
```

---

## **4. Sampling (Reverse Process)**
Iteratively denoise $$ x_T \sim \mathcal{N}(0, \mathbf{I}) $$ to generate $$ x_0 $$.

### **Sampling Steps**
$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t \epsilon_t
$$
- $$ \sigma_t = \sqrt{\frac{(1-\alpha_t)(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}} $$: Stochastic noise term.  
- For $$ t=0 $$, set $$ \epsilon_t = 0 $$.

### **Code Implementation**
```python
def run_inference(unet, num_images):
    x = torch.randn(num_images, 3, 32, 32).to(device)  # x_T ~ N(0,I)
    for t in reversed(range(T)):
        if t > 0:
            z = torch.randn_like(x)  # ε_t
        else:
            z = 0
        
        # Predict noise
        epsilon = unet(x, t_batch)
        
        # Reverse step
        x = (1 / torch.sqrt(alphas[t])) * (x - ((1 - alphas[t]) / torch.sqrt(1 - alpha_bars[t])) * epsilon) \
            + sigma_t[t] * z
    return torch.clamp(x, 0, 1)  # Final x_0
```

---

## **5. Key Takeaways**
1. **Forward Process:** Deterministic noise addition via closed-form Gaussian transitions.  
2. **Reverse Process:** U-Net learns to denoise by predicting $$ \epsilon $$.  
3. **Training:** Simple MSE objective on noise prediction.  
4. **Sampling:** Iterative refinement with stochastic/deterministic steps.  

For clarity on specific diagrams or equations, feel free to ask!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50850777/c4e15dd4-392f-4e4c-b9f8-7fb12b1fda9e/lecture25.pdf

---
Answer from Perplexity: pplx.ai/share