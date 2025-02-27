## AIL 862 - Lecture 7: Target Detection

### Introduction to Target Detection
- Target detection involves identifying and localizing objects in images
- Key challenge: Determining both "what" and "where" in an image

### R-CNN (Region-based Convolutional Neural Network)
- Step 1: Generate category-independent region proposals (~2k per image)
- Step 2: Extract fixed-size feature vectors from each proposal using CNN
- Step 3: Classify regions using class-specific linear SVMs

### Region Proposals
- Methods like Selective Search generate potential object regions
- Proposals can be overlapping and noisy
- Selective Search uses similarity measures: color, texture, size, fill

### Fast R-CNN
- Improves on R-CNN by processing the entire image through CNN once
- Introduces ROI Pooling to convert variable-sized ROIs to uniform size

### Faster R-CNN
- Introduces Region Proposal Network (RPN) to replace traditional proposal methods
- RPN generates proposals directly from feature maps
- Operates in a fully convolutional manner
- Outputs bounding boxes and objectness scores

### Mask R-CNN
- Extends Faster R-CNN for instance segmentation
- Adds a branch for predicting segmentation masks
- Uses ROIAlign for more precise feature extraction

### Applications in Earth Observation
- Oil well detection from high-resolution remote sensing images
- Ship detection in harbors and open sea
- Vehicle detection

### Improvements and Variations
- Feature extraction backbone replacements (e.g., ResNet50)
- Use of dilated convolutions to improve receptive field
- Addition of edge detection modules
- Hard negative mining for improved performance

This lecture covers the evolution of CNN-based object detection methods, from R-CNN to Mask R-CNN, and their applications in Earth Observation tasks. It emphasizes the importance of efficient region proposals and feature extraction techniques for accurate and fast object detection.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/b91a7576-f987-4864-a7e5-a26d9c531cdb/lecture6.pdf
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/bf450669-fe83-48fc-b413-2e5fac791c5d/lecture7.pdf
[3] https://www.mdpi.com/2076-3417/13/12/6913
[4] https://www.mathworks.com/help/vision/ug/object-detection-using-deep-learning.html
[5] https://www.digitalocean.com/community/tutorials/faster-r-cnn-explained-object-detection
[6] https://blog.roboflow.com/mask-rcnn/
[7] https://viso.ai/deep-learning/faster-r-cnn-2/
[8] https://www.scirp.org/journal/paperinformation?paperid=115011
[9] https://paperswithcode.com/method/faster-r-cnn
[10] https://paperswithcode.com/method/mask-r-cnn
[11] https://www.sciopen.com/article/10.26599/BDMA.2024.9020086?issn=2096-0654
[12] https://blog.roboflow.com/what-is-r-cnn/
[13] https://www.kaggle.com/code/silviua/object-detection-cnn
[14] https://ieeexplore.ieee.org/document/7485869
[15] https://viso.ai/deep-learning/mask-r-cnn/
[16] https://www.mdpi.com/journal/remotesensing/special_issues/CNN_RS
[17] https://www.mathworks.com/help/vision/ug/getting-started-with-r-cnn-fast-r-cnn-and-faster-r-cnn.html
[18] https://viso.ai/deep-learning/object-detection/
[19] https://proceedings.neurips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf
[20] https://developers.arcgis.com/python/latest/guide/how-maskrcnn-works/
[21] https://www.researchgate.net/publication/348559309_Review_on_Convolutional_Neural_Networks_CNN_in_Vegetation_Remote_Sensing
[22] https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2022.1095717/full