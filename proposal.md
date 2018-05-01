# CS 230 Project Proposal
## Spring 2018

## Project Title: Brain Tumor Image Segmentation via Capsule Networks
## Project Category: Computer Vision
## Group Member 1: Cam Backes (cbackes@stanford.edu)
## Group Member 2: Jon Deaton (jdeaton@stanford.edu)

### Project Description
Noninvasive methods of brain imaging, most commonly Magnetic Resonance Imaging (MRI), are routinely
used to identify and locate tumors in the brain. Currently, brain tumor image segmentation is a 
time consuming practice which must be performed manually by medical professionals. As such, with
the recent emergence of effective computer vision methods, notably convolutional neural networks 
(CNNs), there is significant practical value in using these tools to automate and improve the 
accuracy of segmentation. We propose using capsule networks to perform segmentation of brain 
tumors in MR images.

### BraTS17 Dataset Description
The Brain Tumor Segmentation (BraTS) dataset was collected by medical professionals from numerous 
institutions including UPenn’s Center for Biomedical Image Computing and Analysis (CBICA), and 
consists of 3D MRI brain scans from 333 individuals with brain tumors, along with age and survival 
information for each individual, and tumor segmentation labels for tumor pixels manually-revised 
by expert board-certified neuroradiologists.

### Algorithmic Approach and Challenges
We have chosen to apply Capsule Network techniques to this problem because of recent advances in 
training Capsule Networks that have shown to improve upon shortcomings of CNNs. The most notable 
difficulty associated with this project is the small size of our dataset. CNNs typically require 
thousands of images for training, so we will need to be particularly cautious not to overfit our 
model to the training data. However, a primary source of our inspiration to use Capsule Networks 
is their demonstrated ability to use training data more efficiently, thus requiring fewer training 
examples than CNNs. Another challenge comes from the fact that MRI scans are three dimensional.

### Relevant Literature
Because this dataset is made publicly accessible for the purpose of promoting growth in the field of 
biomedical imaging analysis, there are numerous relevant papers that elucidate the nature of 
the dataset and explore the use of CNNs and other models for image segmentation. We’ve listed a 
few of these papers in the references section below.

### Result Evaluation
We will evaluate model performance using the metrics defined in [Mense et al.][1], which are used to 
evaluate models submitted to the MICCAI BraTS challenge, an annual competition in which 
researchers’ algorithms are judged based on tumor segmentation performance. For each scan, the 
Dice, sensitivity, and specificity metrics are computed based on discrepancies between predicted 
tumor segmentation, and the true segmentation by doctors. We will evaluate our model by computing 
these metrics using our validation set and comparing our results with those obtained by previous 
challenge participants.

We will display images of brains with our segmentation predictions overlaid, as well as the real 
tumor segmentations for comparison. We will also show histograms of our algorithm’s Dice, 
sensitivity, and specificity scores on the test set versus those of past competition entries, as 
well descriptive statistics that further detail our model’s performance.

### References
Dataset overview:
[The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)][1]

Successful deep learning approaches:
[Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks][2]
[Brain Tumor Segmentation with Deep Neural Networks][3]

[1]:https://www.ncbi.nlm.nih.gov/pubmed/25494501
[2]:https://arxiv.org/abs/1709.00382
[3]:https://arxiv.org/pdf/1505.03540.pdf