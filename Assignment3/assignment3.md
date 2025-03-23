# Assignment 3

Marks: 15

Problem Statement

 - Choose a few distinct image classes of your choice.
 - Use a text-to-image generation model  to generate synthetic images for each class. 
 - Divide your synthetic dataset into two splits - training and validation.
 - Train a deep learning classifier using only the synthetic dataset.
 - Test your classifier on a real dataset consisting of the given classes. Measure performance gap of your model between real dataset and synthetic dataset validation set. 
 - See if this performance gap can be reduced by merely increasing image numbers in the synthetic dataset or some other simple trick during text to image generation.
Consider that you have an unlabeled dataset that mostly has images from the classes of your interest but 10% of this dataset are images from other classes.
Use the above-mentioned unlabeled dataset, potentially with some domain adaptation technique, to further improve the model trained on synthetic data. 
Report Format

Refer to IEEE conference (two column) format, please submit 1-2 page report.

Submission Instruction

In .zip folder (code and report), similar to previous Assignment.

Submission Deadline

March 2, 6 pm



Epoch [8/10], Train Loss: 0.0412, Train Acc: 99.42%, Val Loss: 0.0307, Val Acc: 99.33%
Epoch [9/10], Train Loss: 0.0386, Train Acc: 99.58%, Val Loss: 0.0299, Val Acc: 99.33%
Epoch [10/10], Train Loss: 0.0376, Train Acc: 99.17%, Val Loss: 0.0318, Val Acc: 99.33%
Sample Size: 500, Test Loss: 0.3868, Test Accuracy: 84.55%, Test Error: 15.45%



starting training
Epoch [1/10], Train Loss: 0.2926, Train Acc: 91.28%, Val Loss: 0.0636, Val Acc: 99.44%
Epoch [2/10], Train Loss: 0.0792, Train Acc: 98.14%, Val Loss: 0.0454, Val Acc: 99.44%
Epoch [3/10], Train Loss: 0.0543, Train Acc: 98.73%, Val Loss: 0.0282, Val Acc: 99.44%
Epoch [4/10], Train Loss: 0.0454, Train Acc: 98.98%, Val Loss: 0.0245, Val Acc: 99.86%
Epoch [5/10], Train Loss: 0.0390, Train Acc: 99.02%, Val Loss: 0.0248, Val Acc: 99.44%
Epoch [6/10], Train Loss: 0.0342, Train Acc: 99.02%, Val Loss: 0.0282, Val Acc: 98.87%
Epoch [7/10], Train Loss: 0.0412, Train Acc: 98.77%, Val Loss: 0.0179, Val Acc: 99.58%
Epoch [8/10], Train Loss: 0.0255, Train Acc: 99.44%, Val Loss: 0.0180, Val Acc: 99.58%
Epoch [9/10], Train Loss: 0.0307, Train Acc: 99.19%, Val Loss: 0.0158, Val Acc: 99.58%
Epoch [10/10], Train Loss: 0.0281, Train Acc: 99.19%, Val Loss: 0.0143, Val Acc: 99.58%
Test Loss: 0.3208, Test Accuracy: 87.43%, Test Error: 12.57%


Epoch [1/10], Loss: 0.1162, Accuracy: 95.59%
Epoch [2/10], Loss: 0.0191, Accuracy: 99.35%
Epoch [3/10], Loss: 0.0190, Accuracy: 99.44%
Epoch [4/10], Loss: 0.0187, Accuracy: 99.35%
Epoch [5/10], Loss: 0.0347, Accuracy: 99.05%
Epoch [6/10], Loss: 0.0170, Accuracy: 99.47%
Epoch [7/10], Loss: 0.0098, Accuracy: 99.75%
Epoch [8/10], Loss: 0.0614, Accuracy: 98.74%
Epoch [9/10], Loss: 0.0514, Accuracy: 98.79%
Epoch [10/10], Loss: 0.0206, Accuracy: 99.55%

Final Test Accuracy: 91.62% | Test Loss: 0.6746
