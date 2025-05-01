Here's a detailed breakdown of the **Lecture 18 slides** on Masked Autoencoders (MAE) and their applications:

---

## **Core MAE Architecture**

### **1. Encoder Design**
- **Non-Overlapping Patches:** 
  - Splits images into non-overlapping patches (e.g., 16x16 pixels).
  - **Why?** Overlapping patches introduce redundancy, weakening the learning signal. Non-overlapping patches force the model to infer missing parts without relying on duplicate information[1][6].
- **ViT-Based Encoder:** 
  - Processes only **visible patches** (e.g., 25% of total patches at 75% masking ratio).
  - Uses standard Vision Transformer (ViT) blocks with positional embeddings[6].

### **2. Decoder Design**
- **Input:** 
  - Combines encoded visible patches and **mask tokens** (learnable vectors representing missing patches).
  - Adds positional embeddings to all tokens to preserve spatial information[6][3].
- **Transformer Blocks:** 
  - Lightweight compared to the encoder (reduces pre-training overhead).
  - Reconstructs pixel values for masked patches via a linear projection head[3][5].

### **3. Reconstruction & Loss**
- **Target:** Predicts original pixel values for masked patches.
- **Loss Function:** Mean Squared Error (MSE) between reconstructed and original images[6].

---

## **Key Innovations in MAE**

### **1. High Masking Ratio (75%)**
- **Why 75%?** 
  - Eliminates redundancy, forcing the model to learn global context rather than extrapolating from nearby patches[5][4].
  - Increases the **effective receptive field**, improving feature learning for downstream tasks[2][4].
- **Impact:** 
  - ViT-L pre-trained with MAE achieves **84.9%** ImageNet accuracy vs. **82.5%** for supervised training[1][5].

### **2. Minimal Data Augmentation**
- **Strategy:** Random resized cropping suffices. Color jittering harms performance[5].
- **Comparison:** Unlike contrastive methods (e.g., SimCLR, BYOL), MAE relies on masking as its primary augmentation[5].

| Augmentation       | Fine-Tune Acc | Linear Probe Acc |
|---------------------|---------------|-------------------|
| None                | 84.0%         | 65.7%             |
| Random Resized Crop | **84.9%**     | **73.5%**         |

---

## **Mask Sampling Strategies**

| Strategy | Mask Ratio | Fine-Tune Acc | Linear Probe Acc |
|----------|------------|---------------|-------------------|
| Random   | 75%        | **84.9%**     | **73.5%**         |
| Block    | 75%        | 82.8%         | 63.9%             |
| Grid     | 75%        | 84.0%         | 66.0%             |

- **Random Sampling:** Uniform distribution avoids center bias and maximizes task difficulty[5].
- **Block Sampling:** Removes large contiguous regions, leading to blurry reconstructions[5].
- **Grid Sampling:** Predictable pattern (e.g., remove 1 of 4 patches), less effective for learning[5].

---

## **Applications & Extensions**

### **1. Cross-Domain Adaptations**
- **Microscopy (Channel-Agnostic MAE):**
  - Treats multiple microscopy channels (e.g., fluorescence, brightfield) as a single input.
  - Reconstructs all channels simultaneously[1].
- **Earth Observation (i-MAE):**
  - Processes multi-sensor data (e.g., RGB + infrared) with mixed representations.
  - Uses a shared decoder for multi-modal reconstruction[1].

### **2. Transfer Learning Performance**
- **COCO Object Detection (ViT-L):**
  - MAE pre-training achieves **53.3% AP** vs. **49.3%** for supervised training[1].
- **Linear Probing:** MAE outperforms contrastive methods (e.g., MoCo v3) by **~10%**[5].

---

## **Theoretical Insights**
- **Local Contrastive Learning:** MAE implicitly learns by comparing local patches during reconstruction, with masking ratio controlling the contrastive region size[2].
- **Receptive Field Control:** High masking ratios limit the effective receptive field, preventing overfitting and improving downstream task generalization[2][4].

---

## **Practical Takeaways**
1. **Use Case:** Ideal for scenarios with abundant unlabeled data (e.g., medical imaging, satellite data).
2. **Implementation Tips:**
   - Mask 75% of patches randomly.
   - Use lightweight decoders (e.g., <10% of encoder compute).
   - Minimal augmentation (random crop only).

Let me know if you'd like further clarification on specific diagrams or architectures!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50850777/e5790ff6-4517-4061-9fac-ada1f5a724ab/lecture18.pdf
[2] https://arxiv.org/html/2310.01994v2
[3] https://www.educative.io/courses/mastering-self-supervised-algorithms-for-learning-without-labels/masked-autoencoders-decoder-and-loss-function
[4] https://proceedings.neurips.cc/paper_files/paper/2022/file/adb2075b6dd31cb18dfa727240d2887e-Paper-Conference.pdf
[5] https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf
[6] https://openreview.net/pdf?id=C4wuTlitI7
[7] https://arxiv.org/abs/2202.03670
[8] https://collab.dvb.bayern/spaces/TUMmlneuro/pages/651271074/2+Masked+Autoencoders+Are+Scalable+Vision+Learners
[9] https://arxiv.org/abs/2412.19101
[10] https://arxiv.org/pdf/2206.04846.pdf
[11] https://mchromiak.github.io/articles/2021/Nov/14/Masked-Autoencoders-Are-Scalable-Vision-Learners/
[12] https://docs.lightly.ai/self-supervised-learning/examples/mae.html
[13] https://github.com/Pang-Yatian/Point-MAE/issues/2
[14] https://openaccess.thecvf.com/content/CVPR2023/papers/Bandara_AdaMAE_Adaptive_Masking_for_Efficient_Spatiotemporal_Learning_With_Masked_Autoencoders_CVPR_2023_paper.pdf
[15] https://arxiv.org/abs/2205.14540
[16] https://www.sciencedirect.com/science/article/abs/pii/S0010482523005024
[17] https://www.sciencedirect.com/science/article/abs/pii/S0010482525005104
[18] https://www.ginniemae.gov/issuers/program_guidelines/MBSGuideLib/Chapter_21.pdf
[19] https://www.numberanalytics.com/blog/mae-vs-mse-6-essential-comparisons-in-error-analysis
[20] https://arxiv.org/html/2303.06583v2
[21] https://www.youtube.com/watch?v=fPkp4rlJB6o

---
Answer from Perplexity: pplx.ai/share