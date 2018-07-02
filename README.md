# BraTS18-Project

CS 230 - Deep Learning, Final Project

Stanford University, Spring 2018

##### Cam Backes (cbackes@stanford.edu)
##### Jon Deaton (jdeaton@stanford.edu)

## Automatic Brain Tumor Segmentation
Noninvasive methods of brain imaging, most commonly Magnetic Resonance Imaging (MRI), are routinely used to identify and locate tumors in the brain. Currently, brain tumor image segmentation is a time consuming practice which must be performed manually by medical professionals. As such, with the recent emergence of effective computer vision methods, notably convolutional neural networks (CNNs), there is significant practical value in using these tools to automate and improve the accuracy of segmentation. 

This project explores the application of 3D fully-convolutional deep networks for brain tumor segmentation tasks in magnetic resonance images (MRIs). We created, trained, and tested three variant models of the U-net architecture using the dice coefficient as our primary performance metric to assess the overlap between predicted and ground-truth segmentations. We trained and tested our models using datasets from the [2018 Brain Tumor Segmentation (BraTS) challenge](https://www.med.upenn.edu/sbia/brats2018/data.html), and were able to achieve whole tumor segmentation performance, as indexed by dice score, that is on par with the state-of-the-art from recent years. 

## Model Architecture

We created, trained and evaluated the performance of three deep-convolutional networks using a 3D U-net architecture. The models use three-dimensional convolutions with each of the four image modalities as the input image channels similar to the work of [Cicek et al.](https://arxiv.org/abs/1606.06650). At a high level, each network has three "down" (convolution) blocks that reduce the image spatially through max-pooling, followed by three "up" (transpose convolution) blocks which combine information from the early "down" blocks to increase image dimensions to the match the ground-truth segmentation volume. The number of filters in each convolutional block increases in each successive "down" block until the model reaches its most compact representation at the lowest level with 64 filters. The number of filters then decreases with each successive "up" block until the final output has only a single channel containing the segmentation predictions.

![model_arch](https://user-images.githubusercontent.com/15920014/42143180-b9784310-7d68-11e8-850b-cecf6a3a9175.png)

We used skip connections to propagate relevant spatial information from early layers into the later layers. In an effort to examine trade-offs between model accuracy and efficiency, we trained three models each with different skip connections between "down" and "up" blocks. Our first model uses concatenation of "down" and "up" blocks as skip connections and does not have a dropout layer before the final output. Our second model uses element-wise summations of "down" and "up" blocks as opposed to concatenations for skip connections and includes a dropout layer, while the final model uses neither concatenations nor summations.

## Performance

Models 1 and 2 achieved stellar segmentation performance on the test set, with dice scores of 0.87 and 0.85. The top performing models in recent years' BraTS Challenges have achieved whole tumor dice scores between 0.85 and 0.9, thus making our models' performances on par with the state-of-the-art. We believe that model 1's marginally superior performance is due to the enhanced access to important spatial information provided by the concatenations.


Table 1: Mean and standard deviation of dice coefficients for brain tumor example from the training, test, and validation sets, respectively.

| Model         | Train (n=204)   | Test (n=40)    | Validation (n=40)   |
|---------------|-----------------|----------------|---------------------|
| Concatenation | 0.907 +/- 0.047 | 0.87 +/- 0.072 | 0.89 +/- 0.07       |
| Summation     | 0.89 +/- 0.05   | 0.85 +/- 0.09  | 0.87 +/- 0.08       |
| No-skip       | 0.85 +/- 0.13   | 0.81 +/- 0.12  | 0.77 +/- 0.25       |


Table 2: Minimum and maximum dice coefficients for brain tumor example from the training, test, and validation sets, respectively.

| Model         | Train (n=204)               | Test (n=40)                | Validation (n=40)         |
|---------------|-----------------------------|----------------------------|---------------------------|
| Concatenation | min  = 0.720,  max  = 0.96  | min  = 0.67,  max  = 0.96  | min = 0.71, max = 0.96    |
| Summation     | min  = 0.62,  max  = 0.97   | min  = 0.48,  max  = 0.96  | min =  0.62 , max = 0.97  |
| No-skip       | min  = 0.13,  max  =0.96    | min  =0.43,  max  = 0.95   | min = 3.7E-7 , max =0.95  |


### Installation

Install the required dependencies

    python setup.py install

### Usage

To train the model

    python -m segmentation.train
    
To create TFRecord files

    python -m preprocessing.createTFRecords --brats ~/Datasets/BraTS/ --year 2018 --output ~/Datasets/BraTS/TFRecords

In order to ake sure that you are only using 1 GPU:
    
    export CUDA_VISIBLE_DEVICES=1


## BraTS Data Loader

This package comes with a data-loader package which provides convenient programmatic access to the BraTS dataset through a python module. This module abstracts away the organization of the data on the file system and file formats that store the MRI images and meta-data. In order to use the data loader make sure that your BraTS dataset directory is configured as shown:

    BraTS
    ├── BraTS15
    │   ├── training
    │   └── validation
    ├── BraTS17
    │   ├── training
    │   └── validation
    └── BraTS18
        └── training
 
 
 You can import the BraTS data-loader into Python
 
     import BraTS
 
 and configure it to find the BraTS data sets on your system by passing the top level directroy plus year
 
     brats = BraTS.DataSet(brats_root="/path/to/BraTS", year=2017)
 
 or explicitly providing a path to a BraTS directory
 
    brats = BraTS.DataSet(data_set_dir="/path/to/BraTS/BraTS17")
 
Then, you can access it's data members through `train`, `validation`, `hgg` and `lgg` members
 
    # Access data patient-wise
    patient = brats.train.patient("Brats18_2013_7_1")   # Loads only a single patient (quick)
    
    # Iterate through all patients
    for patient in brats.train.patients:
        process(patient.mri, patient.seg) # example
       
    # Access to all the patient IDs
    patient_ids = brats.train.ids
 
    mris = brats.train.mris  # (m, 4, 240, 240, 155) numpy array of all MRI images
    segs = brats.train.segs  # (m, 240, 240, 255) numpy array of segmentation maps
    
    # Access validation data like this
    validation_mris = brats.validation.mris
    validation_segs = brats.validation.segs
     
 Be aware that it takes a few minutes to load the full data-set into memory. If you want to load
 in only single patients at a time, then use the `brats.train.patient` interface.
 
