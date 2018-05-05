# BraTS18-Project

##### Cam Backes (cbackes@stanford.edu)
##### Jon Deaton (jdeaton@stanford.edu)

## Brain Tumor Segmentation
Noninvasive methods of brain imaging, most commonly Magnetic Resonance Imaging (MRI), are routinely
used to identify and locate tumors in the brain. Currently, brain tumor image segmentation is a 
time consuming practice which must be performed manually by medical professionals. As such, with
the recent emergence of effective computer vision methods, notably convolutional neural networks 
(CNNs), there is significant practical value in using these tools to automate and improve the 
accuracy of segmentation. We propose using capsule networks to perform segmentation of brain 
tumors in MR images.




#### BraTS Data Loader


In order to use the data loader make sure that your BraTS dataset directory is configured as shown:

BraTS
├── BraTS15
│   ├── BraTS15_Training
│   └── BraTS15_Validation
├── BraTS17
│   ├── BraTS17_Training
│   └── BraTS17_Validation
└── BraTS18
    └── BraTS18_Training
 
 
 You can import the BraTS data-loader into Python
 
     import BraTS
 
 and configure it to find the BraTS data sets on your system by passing the path to the top level directory with the `set_root` function:
 
     BraTS.set_root("BraTS")
 
 To access the data, first select a data-set by year
 
     brats = BraTS.DataSet(year=2018)
 
 and then access it's data members through `train`, `validation`, `hgg` and `lgg` members
 
    mris = brats.train.mris  # (m, 4, 240, 240, 155) numpy array of all MRI images
    segs = brats.train.segs  # (m, 240, 240, 255) numpy array of segmentation maps
    
    # Access data patient-wise
    patient = brats.train.patient("Brats18_2013_7_1")
    patients = brats.train.patients["Brats18_2013_7_1"]
    
    # 
    
 Be aware that it takes a few minutes to load the dataset into memory.
 