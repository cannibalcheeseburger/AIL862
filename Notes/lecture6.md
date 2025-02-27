## AIL 862 - Lecture 6: Advanced Semantic Segmentation Architectures

### Challenges in Semantic Segmentation

- Determining both "what" and "where" in an image
- Distinguishing classes with similar spectral signatures
- Identifying inconspicuous classes
- Handling object size variations within images[1]
- Dealing with confused object classes[1]

### Solutions to Segmentation Challenges

- Incorporating context at multiple scales
- Leveraging global context
- Using dense skip connections and adaptive fusion attention modules[1]
- Employing context aggregation and attention mechanisms[1]

### U-Net Architecture

U-Net is a popular architecture for semantic segmentation, particularly in biomedical imaging[8]. Key features include:

- Contraction and expansion phases
- Skip connections between contraction and expansion paths
- Ability to work with limited training data through extensive data augmentation[15]

### Advanced Architectures

#### PSPNet (Pyramid Scene Parsing Network)

- Uses pyramid pooling module to exploit global context
- Aggregates context at different scales
- Combines local and global information for more reliable predictions[9]

#### DeepLabv3+

- Employs atrous (dilated) convolutions
- Features Atrous Spatial Pyramid Pooling (ASPP) module
- Captures multi-scale context effectively[10]

### Specialized Applications

#### SAR (Synthetic Aperture Radar) Segmentation

- Applies similar architectures as optical image segmentation
- Often yields suboptimal performance compared to optical imagery
- Researchers explore additional training data sources or more reliable label identification techniques[13]

#### Hyperspectral Image Segmentation

- Utilizes channel attention mechanisms
- Addresses the high dimensionality of hyperspectral data[12]

### Future Trends

- Integration of transformer-based models like Vision Transformers (ViTs)[14]
- Development of more efficient architectures like EfficientNet for resource-constrained environments[14]
- Increasing use of multi-modal data fusion, combining RGB images with depth or thermal data[14]
- Focus on real-time semantic segmentation for applications like autonomous driving[14]

This lecture covers advanced architectures for semantic segmentation, focusing on techniques to address challenges like multi-scale context, global information integration, and efficient feature extraction. It also touches on applications in specialized domains like SAR and hyperspectral imaging, highlighting the ongoing research to improve performance in these areas.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/50850777/b91a7576-f987-4864-a7e5-a26d9c531cdb/lecture6.pdf
[2] https://labelyourdata.com/articles/data-annotation/semantic-segmentation
[3] https://www.ibm.com/think/topics/semantic-segmentation
[4] https://www.frontiersin.org/journals/remote-sensing/articles/10.3389/frsen.2024.1370697/full
[5] https://dl.gi.de/bitstreams/40dbabde-8332-42aa-82f8-bda5b5290768/download
[6] https://keylabs.ai/blog/how-to-improve-accuracy-in-semantic-segmentation/
[7] https://openaccess.thecvf.com/content_ECCV_2018/html/Di_Lin_Multi-Scale_Context_Intertwining_ECCV_2018_paper.html
[8] https://paperswithcode.com/method/u-net
[9] https://wiki.cloudfactory.com/docs/mp-wiki/model-architectures/pspnet
[10] https://github.com/mukund-ks/DeepLabV3-Segmentation
[11] https://www.mdpi.com/2072-4292/15/8/2153
[12] https://www.mdpi.com/journal/electronics/special_issues/HI_electronics
[13] https://www.worldscientific.com/doi/10.1142/S0218001422520279
[14] https://clarion.ai/the-future-of-semantic-segmentation/
[15] https://viso.ai/deep-learning/u-net-a-comprehensive-guide-to-its-architecture-and-applications/
[16] https://www.nature.com/articles/srep38596
[17] https://www.superannotate.com/blog/guide-to-semantic-segmentation
[18] https://paperswithcode.com/task/semantic-segmentation
[19] https://paperswithcode.com/task/semantic-segmentation/latest
[20] https://viso.ai/deep-learning/semantic-segmentation-instance-segmentation/
[21] https://www.nature.com/articles/s41598-024-71080-4
[22] https://www.researchgate.net/publication/377933146_A_review_on_current_progress_of_semantic_segmentation
[23] https://viso.ai/deep-learning/panoptic-segmentation-a-basic-to-advanced-guide-2024/
[24] https://www.mathworks.com/solutions/image-video-processing/semantic-segmentation.html
[25] https://ieeexplore.ieee.org/document/10750790
[26] https://keylabs.ai/blog/advanced-techniques-in-semantic-segmentation/
[27] https://www.byteplus.com/en/topic/100761
[28] https://neptune.ai/blog/image-segmentation
[29] https://www.linkedin.com/pulse/semantic-segmentation-deep-dive-cutting-edge-computer-vision-elf8c
[30] https://encord.com/blog/guide-to-semantic-segmentation/
[31] https://ieeexplore.ieee.org/document/8936087/
[32] https://www.researchgate.net/figure/Examples-of-challenges-for-semantic-segmentation-We-show-the-original-images-1st-row_fig1_329021980
[33] https://arxiv.org/abs/2406.05837
[34] https://ieeexplore.ieee.org/document/10530477/
[35] https://www.cloudfactory.com/blog/five-segmentation-challenges-in-machine-learning
[36] https://www.cs.huji.ac.il/~danix/publications/MSCI.pdf
[37] https://www.mdpi.com/2079-9292/12/5/1199
[38] https://www.mdpi.com/2072-4292/12/4/701
[39] https://arxiv.org/abs/2203.00214
[40] https://www.nature.com/articles/s41598-024-66585-x
[41] https://www.linkedin.com/advice/0/what-most-common-challenges-image-segmentation-9spmf
[42] https://ieeexplore.ieee.org/document/10462518/
[43] https://en.wikipedia.org/wiki/U-Net
[44] https://www.wiz.ai/link-net-doubles-its-customer-outreach-with-wiz-talkbots/
[45] https://www.kaggle.com/code/michaelcripman/image-segmentation-using-pspnet
[46] https://learnopencv.com/kerascv-deeplabv3-plus-semantic-segmentation/
[47] https://www.digitalocean.com/community/tutorials/unet-architecture-image-segmentation
[48] https://www.theobjects.com/dragonfly/dfhelp/Content/Resources/PDFs/linknet.pdf
[49] https://ieeexplore.ieee.org/document/9952233/
[50] https://keras.io/examples/vision/deeplabv3_plus/
[51] https://www.nature.com/articles/s41598-023-34379-2
[52] https://ieeexplore.ieee.org/document/9944630/
[53] https://www.researchgate.net/figure/mage-segmentation-effect-of-PSPNet-model_fig6_372149132
[54] https://www.ikomia.ai/blog/understanding-deeplabv3-image-segmentation
[55] https://www.earthdata.nasa.gov/learn/earth-observation-data-basics/sar
[56] https://www.mdpi.com/2072-4292/12/5/803
[57] https://ieeexplore.ieee.org/document/10037590/
[58] https://ieeexplore.ieee.org/iel8/4609443/10330207/10590728.pdf
[59] https://www.researchgate.net/publication/365781955_Automated_SAR_Image_Segmentation_and_Classification_using_Modified_Deep_Learning
[60] https://www.researchgate.net/publication/364883749_Hyperspectral_image_segmentation_a_comprehensive_survey
[61] https://www.researchgate.net/publication/3357720_Segmentation-based_joint_classification_of_SAR_and_optical_images
[62] https://ieeexplore.ieee.org/iel8/4609443/10330207/10547107.pdf
[63] https://ieeexplore.ieee.org/document/7326632/
[64] https://www.researchgate.net/publication/382138849_A_review_of_optical_and_SAR_image_deep_feature_fusion_in_semantic_segmentation
[65] https://arxiv.org/html/2502.12541v1
[66] https://arxiv.org/pdf/1901.03749.pdf
[67] https://keylabs.ai/blog/cutting-edge-semantic-segmentation-algorithms/
[68] https://keymakr.com/blog/exploring-the-top-algorithms-for-semantic-segmentation/
[69] https://www.linkedin.com/advice/0/what-some-latest-advances-challenges-semantic
[70] https://arxiv.org/html/2404.16573v1
[71] https://openaccess.thecvf.com/content_ICCV_2019/papers/He_Dynamic_Multi-Scale_Filters_for_Semantic_Segmentation_ICCV_2019_paper.pdf
[72] https://www.tableau.com/solutions/customer/link-net-realizes-benefits-self-service-analytics-with-tableau
[73] http://cic.tju.edu.cn/faculty/zhangkl/web/aboutme/apweb05-draft-paper.pdf
[74] https://developers.arcgis.com/python/latest/guide/how-pspnet-works/
[75] https://www.datature.io/blog/a-guide-to-using-deeplabv3-for-semantic-segmentation
[76] https://www2.umbc.edu/rssipl/people/aplaza/Papers/Journals/2007.RSE.Advances.pdf
[77] https://www.mdpi.com/2072-4292/15/3/850
[78] https://pmc.ncbi.nlm.nih.gov/articles/PMC3280832/
[79] https://www.mdpi.com/2076-3417/14/11/4909