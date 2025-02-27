Okay, I've gone through the slides for Lecture 3 and will provide detailed notes as a student, including additional information from the internet.

## AIL 862 - Lecture 3: Challenges in Supervised Learning and Domain Adaptation

**Slide 1: AIL 862, Lecture 3**

*   Lecture number and course code.

**Slide 2: Usual Recipe (Recap)**

*   Reiterating the CNN training process:
    1.  Understand task (domain, classes).
    2.  Collect images.
    3.  Annotate images.
    4.  Train CNN model.
    5.  Deploy.
*   The key question: "Does this recipe always work?" - This implies we're moving into the limitations and challenges.

**Slide 3: Label - Automobile, Bird**

*   Examples from the CIFAR-10 dataset.
*   **Note:** CIFAR-10 is a standard dataset with 60,000 32x32 color images in 10 classes. Good for initial experimentation and benchmarking.
*   **Point:** These labels are relatively straightforward and unambiguous.

**Slide 4: Label - Building, also road, cars**

*   Example from the UC Merced dataset.
*   **Note:** UC Merced dataset is used for land use classification with aerial imagery.
*   **Key Point:** This highlights ambiguity - a single image can contain multiple objects and potentially multiple correct labels.

**Slide 5: Annotation: difficulty and ambiguity**

*   Annotation requires domain knowledge.
*   **Example:** Accurately annotating medical images requires medical expertise. Similarly, interpreting satellite imagery requires understanding of remote sensing principles.
*   **Takeaway:** High-quality annotations are crucial for supervised learning, and domain expertise is often essential for achieving this.

**Slide 6: Weak Supervision**

*   Example of OSM (OpenStreetMap)-based building annotation.
*   **Concept:** Using readily available, but potentially noisy, data for annotation.
*   **Note:** OSM provides a wealth of geographic data, but its accuracy and completeness can vary.

**Slide 7: Weak Supervision**

*   Weak supervision is often misaligned.
*   **Example:** Georeferencing issues with OSM data can lead to inaccurate building annotations.
*   **Challenge:** Dealing with noisy or inaccurate labels in training data.

**Slide 8: Multi-source/sensor/modality data**

*   Key question: "Challenge or complementary information?"
*   This is where we start to consider integrating data from multiple sources.

**Slide 9: Improving temporal resolution using multisensor information**

*   Example of combining data from different sensors to improve temporal resolution.
*   **Context:** Combining high spatial resolution imagery (e.g., from satellites with infrequent revisits) with more frequent, lower spatial resolution data to track changes over time.

**Slide 10: Multi-Sensor EO**

*   Challenges of multi-sensor EO:
    *   Differences in spatial and temporal resolution.
    *   Differences in spectral characteristics.
*   **Implication:** Need methods to fuse or harmonize data from different sensors.

**Slide 11: Multi-modality**

*   Example: Building type classification using images + tweets.
*   **Challenge:** Geographic localization of non-EO data.
*   **Question:** Can different modalities be processed using similar architectures?
*   **Note:** This relates to multimodal learning, where models learn from multiple data types (e.g., images, text, audio).

**Slide 12: Multi-sensor Medical Image Analysis**

*   Example from medical imaging.

**Slide 13: Multi-modality in medical image processing**

*   Medical images + text data + patient history.
*   **Opportunity:** Combining different data sources to improve diagnosis and treatment planning.

**Slide 14: Domain Differences**

*   Introduction to the concept of *domain shift*.
*   Models trained on one domain may not generalize well to other domains.

**Slide 15: DomainNet dataset**

*   Examples of different domains within the DomainNet dataset:
    *   Clipart
    *   Real
    *   Sketch
    *   Infograph
    *   Painting
    *   Quickdraw
*   **Takeaway:** Images from different domains can have drastically different visual characteristics.

**Slide 16: Domain Differences**

*   Examples of domain differences in Earth Observation:
    *   Concept
    *   Sensor
    *   Season
    *   Geography

**Slide 17: Domain Differences**

*   Relating domain differences to autonomous driving (sunny, rainy, foggy conditions).
*   Reinforcing the idea that models must be robust to variations in the input data.

**Slide 18: OOD (Out-of-Distribution)**

*   Introduction to the concept of Out-of-Distribution (OOD) data.

**Slide 19: Two options**

*   Two approaches to dealing with OOD data:
    *   Identify and leave out OOD domains.
    *   Generalize to OOD domains.

**Slide 20: Big Noisy Data**

*   Visual data is continuously collected, but often challenging to annotate.
*   Many implicit labeling mechanisms available.
*   **Example:** Using social media data or sensor logs as a form of weak supervision.

**Slide 21: Supervised Learning: 4 components**

*   The fundamental components of supervised learning:
    1.  Training data
    2.  Learner
    3.  Learning algorithm
    4.  Performance

**Slide 22: Supervised Learning: 4 components (Detailed)**

*   Expanding on the components:
    *   Training data: Input features, target/feedback.
    *   Learner: Parameters θ.
    *   Learning algorithm: Changes the parameters and improves performance.
    *   Performance: Cost function.
*   The learning cycle: Predict - Score - Learn - ...

**Slide 23: Deep Learning**

*   Analogy to our multi-year education.
*   **Implication:** Deep learning models learn hierarchical representations of data, similar to how we build knowledge over time.

**Slide 24: Supervised Learning: various issues**

*   Issues with supervised learning:
    *   Dependence on training data.
    *   Overfitting and complexity (complex models overfit).
    *   Choice of (hyper)parameters.

**Slide 25: Supervised Learning: Learning Rate**

*   Starting from pretrained model and showing that a good learning rate is needed.

**Slide 26: Supervised Learning: Learning Rate**

*   Starting from pretrained model and showing that with low learning rate performance is not good

**Slide 27: Supervised Learning: Learning Rate**

*   Starting from pretrained model and showing that high learning rate is also not good

**Slide 28: Data Split**

*   Standard data split for machine learning:
    *   Training
    *   Validation
    *   Test

**Slide 29: Supervised Learning**

*   Visual representation of supervised learning

**Slide 30: Training Data Amount**

*   The effect of Training Data Amount on model performance is shown.

**Slide 31: Using Models Trained in Supervised Fashion For Some Other Task**

*   Three ways to leverage pre-trained models:
    *   Just use as feature extractor.
    *   Fine-tuning on target data.
    *   Unsupervised domain adaptation.

**Slide 32: Just as Feature Extractor**

*   Illustration of using a pre-trained model as a feature extractor.

**Slide 33: Some Aspects**

*   Some considerations when using a pre-trained model as a feature extractor.

**Slide 34: Using Models Trained in Supervised Fashion For Some Other Task**

*   Three ways to leverage pre-trained models:
    *   Just use as feature extractor.
    *   Fine-tuning on target data.
    *   Unsupervised domain adaptation.

**Slide 35: Fine Tuning**

*   Illustration of fine tuning, where some layers are frozen and others are trained.

**Slide 36: Fine Tuning**

*   Comparison of training from ImageNet-trained model versus training from scratch.

**Slide 37: Fine tuning**

*   "Pre-training + fine-tuning does not always work."
*   Initial layers are often domain-specific.

This lecture provides a solid overview of the challenges in supervised learning, particularly in the context of image analysis and Earth Observation. The importance of data quality, domain knowledge, multi-source data integration, and domain adaptation are highlighted. The discussion on pre-trained models and fine-tuning offers practical strategies for leveraging existing knowledge.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/2c20c9c1-9d8e-4854-9bf9-c918a289875c/lecture1.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/009a6acc-4546-4cfc-8dea-4094b7558309/lecture2.pdf
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/a2f3fee4-c1a4-447c-a1a7-8fef23bae75a/lecture3.pdf