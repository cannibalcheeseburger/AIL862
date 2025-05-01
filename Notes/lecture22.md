## Detailed Explanation of Lecture 22 Slides: Target Presence Detection & Multi-Temporal Change Detection

---

### **1. Target Presence Detection**

#### **Objective**
Determine whether a specific object or feature exists in an image (binary "yes/no"), **without locating it precisely**.  
**Applications:** Agriculture (detecting crop stress), disaster management (identifying flood-affected areas), planetary exploration.

#### **Methodology**
- **Unsupervised Learning:** Uses only **one example image** of the target object.
- **SAM-Based Workflow:**
  1. **Stitch** the query image with example images (prompts) to form a composite input.
  2. Run SAM multiple times with **different prompts** (points, boxes) on the query image.
  3. Extract **confidence scores** (IoU predictions) for each prompt.
  4. Compare the **highest confidence score** against a threshold to decide presence.

**Challenges:**
- **Spatial Bias:** SAM may focus on the example half of stitched images, leading to false negatives.  
- **Threshold Tuning:** Requires domain-specific calibration for accuracy.

---

### **2. Multi-Temporal Change Detection**

#### **Objective**
Identify changes between images of the same location captured at different times.  
**Applications:**  
- Environmental monitoring (deforestation).  
- Urban planning (infrastructure growth).  
- Disaster response (earthquake damage assessment).

#### **Sensors & Data**
- **Optical:** Multispectral (Sentinel-2), hyperspectral.  
- **SAR:** Synthetic Aperture Radar (all-weather imaging).  
- **Resolution:** Varies from meters (Sentinel-2) to sub-meter (VHR satellites).

---

### **3. Traditional vs. Deep Learning Methods**

| **Method**               | **Approach**                                      | **Limitations**                              |
|--------------------------|---------------------------------------------------|----------------------------------------------|
| **Change Vector Analysis (CVA)** | Pixel-wise differencing of spectral bands.       | Ignores spatial context; sensitive to noise. |
| **Object-Based Clustering** | Groups pixels into super-pixels for comparison.  | Struggles with high-resolution data.         |
| **Deep CVA (2018)**       | Uses CNNs to extract features for bi-temporal differencing. | Requires sensor-specific tuning.             |

#### **Deep CVA Workflow**
1. **Input:** Pre-processed bi-temporal images (same sensor).  
2. **Feature Extraction:** CNN encodes spatial-temporal patterns.  
3. **Differencing:** Computes deep feature differences (Δ = |F₁ - F₂|).  
4. **Output:**  
   - **Binary Change Map:** Labels pixels as changed/unchanged.  
   - **Multi-Class Change Map:** Categorizes change types (e.g., deforestation vs. construction).

**Performance on Sentinel-2:**
- Outperforms traditional CVA in detecting urban expansion and vegetation loss.  
- Struggles with **cross-sensor generalization** (e.g., optical → SAR).

---

### **4. Challenges in Change Detection**
- **Data Scarcity:** Labeled multi-temporal datasets are rare.  
- **Sensor Variability:** Models trained on optical data underperform on SAR.  
- **Temporal Misalignment:** Seasonal variations (e.g., snow cover) can cause false positives.

---

### **Key Takeaways**
1. **Target Presence Detection:** SAM’s prompt-based approach enables zero-shot detection but requires careful thresholding.  
2. **Change Detection:**  
   - **Traditional Methods:** Simple but lack context.  
   - **Deep Learning:** Captures complex patterns but needs sensor-specific adaptation.  
3. **Future Directions:**  
   - Few-shot learning for cross-sensor generalization.  
   - Integration of SAM for interactive change annotation.  

Let me know if you’d like a deeper dive into specific workflows or code implementations!

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/50850777/68c2ebdc-06ea-4a97-a33f-e03b14d78af5/lecture22.pdf

---
Answer from Perplexity: pplx.ai/share