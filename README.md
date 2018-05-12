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
 
