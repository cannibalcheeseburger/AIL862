Okay, here are my detailed notes specifically for Lecture 4, enhanced with external information and context.

## AIL 862 - Lecture 4: Advanced Fine-Tuning and Architectural Insights

**Slide 1: AIL 862, Lecture 4**

*   Standard lecture header.

**Slide 2: Fine Tuning (Recap)**

*   Further train particular layers of the network, with some layers frozen.
*   **Relevance:** This builds upon the previous lecture's discussion of leveraging pre-trained models.
*   **Reminder:** Freezing layers prevents the pre-trained weights from changing, allowing the model to adapt only the unfrozen layers to the new task. This is particularly useful when the target dataset is small.

**Slide 3: Fine tuning**

*   "Pre-training + fine-tuning does not always work." This is a critical point.
*   Initial layers are often domain-specific.
*   **Explanation:** Early layers in a CNN learn low-level features (edges, corners, textures) that might be specific to the domain the model was originally trained on (e.g., ImageNet).
*   **Note:** Transferring a model trained on natural images directly to satellite imagery might not be optimal because the features required to recognize objects in these images are different.
*   **Question:** How do we determine which layers to freeze/fine-tune? Trial and error, validation performance, and understanding the similarity between the source and target domains are key.

**Slide 4: Fine tuning**

*   Training UC Merced (optical dataset) starting from a SAR-trained model might not be optimal.
*   **Explanation:** SAR (Synthetic Aperture Radar) data is fundamentally different from optical imagery. SAR provides information about surface roughness and dielectric properties, while optical sensors measure reflected sunlight.
*   **Takeaway:** Carefully consider the compatibility of the source and target domains when using fine-tuning. Starting with a model pre-trained on a more relevant domain will generally yield better results.

**Slide 5: Using Models Trained in Supervised Fashion For Some Other Task (Recap)**

*   Three approaches to leverage pre-trained models:
    1.  Just use as feature extractor.
    2.  Fine-tuning on target data.
    3.  Unsupervised domain adaptation.
*   **Relevance:** Setting the stage for a deeper dive into unsupervised domain adaptation.

**Slide 6: Unsupervised Domain Adaptation**

*   **Concept:** Adapting a model to a new domain (target domain) without labeled data in the target domain, using only labeled data from the source domain.
*   **Note:** This is a challenging but highly valuable technique for scenarios where annotating data in the target domain is expensive or impossible.

**Slide 7: Unsupervised Domain Adaptation**

*   Batch Normalization-based Domain Adaptation: Aligning feature distributions through feature standardization. Setting the mean of features to 0 and variance to 1.
*   **Explanation:** Batch Normalization (BN) helps to stabilize training by normalizing the activations within each batch. In domain adaptation, BN can be used to reduce the distribution shift between the source and target domains.
*   **BN Adaptation Methods:** Techniques like *Test-Time Adaptation* and *Adaptive Batch Normalization* adjust the BN statistics using unlabeled target data during testing.

**Slide 8: Batch Normalization**

*   During training, running mean and variance are estimated.
*   For domain adaptation, it's suggested to estimate these statistics from test time minibatches (containing target domain data).
*   **Note:** Standard BN uses statistics from the training data, which might not be representative of the target domain. Estimating the statistics from target domain data can improve generalization.

**Slide 9: Unsupervised Domain Adaptation**

*   Domain Translation (GANs?)
*   **Question:** Why aren't GANs used as much for this?
*   **Possible Reasons:**
    *   GANs can be unstable to train.
    *   Mode collapse (the generator produces a limited variety of images).
    *   Difficulty in preserving content information during translation.

**Slide 10: Unsupervised Domain Adaptation**

*   Domain Confusion / Domain Adversarial Training
*   **Concept:** Training a domain classifier to distinguish between the source and target domains while simultaneously training the feature extractor to confuse the domain classifier. This encourages the feature extractor to learn domain-invariant features.
*   **Note:** The Domain Adversarial Neural Network (DANN) is a popular example of this approach.

**Slide 11: LeNet**

*   LeNet architecture diagram.
*   **Context:** LeNet was one of the earliest CNN architectures, designed for handwritten digit recognition.
*   **Relevance:** Provides a historical perspective and highlights the evolution of CNN architectures.

**Slide 12: ImageNet Large Scale Visual Recognition Challenge 2010 (ILSVRC2010)**

*   Details about the ILSVRC2010 challenge.
*   1000 object categories.
*   **Relevance:** ILSVRC was a major driving force behind the advancements in deep learning for computer vision.
*   **Note:** The challenge involved classifying images into one of 1000 categories.
*   **Key takeaway:** the results from the challenge highlight the significant performance leap achieved with deep learning methods compared to traditional approaches.

**Slide 13: AlexNet**

*   AlexNet architecture diagram.
*   **Significance:** AlexNet was a breakthrough CNN architecture that won the ILSVRC2012 competition by a large margin.
*   **Key features:**
    *   Deeper than previous networks (5 convolutional layers and 3 fully connected layers).
    *   ReLU activation function.
    *   Dropout for regularization.
    *   Data augmentation.
    *   GPU acceleration.

**Slide 14: AlexNet**

*   The architecture was split across two GPUs due to memory limitations.

**Slide 15: AlexNet**

*   Five convolutional and three fully connected layers.
*   The first convolutional layer uses 96 kernels of spatial size 11x11. These large filters allowed the network to capture complex and high-level features from images.

**Slide 16: AlexNet further used**

*   Dropout
*   ReLU
*   Max Pooling
*   Data augmentation

**Slide 17: VGGNet**

*   Unlike AlexNet, VGGNet promoted the use of 3x3 convolution filters.
*   Maintaining a consistent convolution filter size across all layers and increasing network depth improved performance.
*   VGG nets uses smaller and multiple filters of size 3x3.

**Slide 18: VGGNet**

*   VGGNet has more layers and a larger number of parameters compared to previous models but requires fewer epochs for training.
*  Smaller filter sizes and pre-initialized weights contribute to training efficiency.

**Slide 19: Inception Module**

*   The architecture diagram of the Inception Module is showcased.

**Slide 20: GoogleNet**

*   InceptionModule inspired by "intuition of multiscale processing"
*   Uses inception module many times over the network

**Slide 21: Project depth to lower dimension**

*   Uses 1x1 Convolution filter.

**Slide 22: Inception Module (Revised)**

*   The architecture diagram of the Inception Module (Revised) is showcased.

In summary, Lecture 4 delves into advanced techniques for leveraging pre-trained models, including fine-tuning strategies and unsupervised domain adaptation. The lecture emphasizes the importance of understanding domain differences and selecting appropriate adaptation techniques. Additionally, a review of influential CNN architectures (LeNet, AlexNet, VGGNet, and GoogleNet) provides valuable context for understanding the evolution of deep learning for computer vision.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/2c20c9c1-9d8e-4854-9bf9-c918a289875c/lecture1.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/009a6acc-4546-4cfc-8dea-4094b7558309/lecture2.pdf
[3] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/a2f3fee4-c1a4-447c-a1a7-8fef23bae75a/lecture3.pdf
[4] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/c45a81ef-01c5-4c01-9199-43d41cfce21c/lecture4.pdf
