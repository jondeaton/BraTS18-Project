#!/usr/bin/env python
"""
File: convert-dataset
Date: 5/5/18

----------------------------------------

Tools for converting, normalizing, and fixing the brats data.
"""

import glob
import os
import warnings
import shutil

import numpy as np
import SimpleITK as sitk  # If you can't import this then run "conda install -c simpleitk simpleitk"
from nipype.interfaces.ants import N4BiasFieldCorrection

from BraTS.modalities import Modality, get_modality_map, modalities, modality_names


def get_background_mask(input_dir, out_file):
    """
    This function computes a common background mask for all of the data in a subject folder.
    :param input_dir: a subject folder from the BRATS dataset.
    :param out_file: an image containing a mask that is 1 where the image data for that subject contains the background.
    :param truth_name: how the truth file is labeled int he subject folder
    :return: the path to the out_file
    """
    background_image = None

    modality_map = get_modality_map(input_dir)
    for modality in modalities:
        image_file = modality_map[modalities]
        image = sitk.ReadImage(image_file)
        if background_image:
            if modality == Modality.seg:
                image.SetOrigin(background_image.GetOrigin())
            background_image = sitk.And(image == 0, background_image)
        else:
            background_image = image == 0
    sitk.WriteImage(background_image, out_file)
    return os.path.abspath(out_file)


def convert_image_format(in_file, out_file):
    sitk.WriteImage(sitk.ReadImage(in_file), out_file)
    return out_file


def window_intensities(in_file, out_file, min_percent=1, max_percent=99):
    image = sitk.ReadImage(in_file)
    image_data = sitk.GetArrayFromImage(image)
    out_image = sitk.IntensityWindowing(image,
                                        np.percentile(image_data, min_percent),
                                        np.percentile(image_data, max_percent))
    sitk.WriteImage(out_image, out_file)
    return os.path.abspath(out_file)


def correct_bias(in_file, out_file):
    """
    Corrects the bias using ANTs N4BiasFieldCorrection.

    If this fails, will then attempt to correct bias using SimpleITK
    :param in_file: input file path
    :param out_file: output file path
    :return: file path to the bias corrected image
    """
    correct = N4BiasFieldCorrection()
    correct.inputs.input_image = in_file
    correct.inputs.output_image = out_file
    try:
        done = correct.run()
        return done.outputs.output_image
    except IOError:
        warnings.warn(RuntimeWarning("ANTs N4BIasFieldCorrection could not be found."
                                     "Will try using SimpleITK for bias field correction"
                                     " which will take much longer. To fix this problem, add N4BiasFieldCorrection"
                                     " to your PATH system variable. (example: EXPORT ${PATH}:/path/to/ants/bin)"))
        output_image = sitk.N4BiasFieldCorrection(sitk.ReadImage(in_file))
        sitk.WriteImage(output_image, out_file)
        return os.path.abspath(out_file)


def rescale(in_file, out_file, minimum=0, maximum=20000):
    image = sitk.ReadImage(in_file)
    sitk.WriteImage(sitk.RescaleIntensity(image, minimum, maximum), out_file)
    return os.path.abspath(out_file)


def background_to_zero(in_file, background_file, out_file):
    sitk.WriteImage(sitk.Mask(sitk.ReadImage(in_file), sitk.ReadImage(background_file, sitk.sitkUInt8) == 0),
                    out_file)
    return out_file


def get_output_filename(output_dir, modality):
    return os.path.join(output_dir, "%s.nii.gz" % modality_names[modality])


def normalize_patient_images(patient_dir, output_patient_dir):
    """
    Corrects the bias for the images in a single patient directory

    :param patient_dir: The patient directory
    :param output_patient_dir: Output directory to store corrected images
    :return: None
    """
    modality_map = get_modality_map(patient_dir)
    for modality in modalities:

        if modality == Modality.seg:
            continue  # don't normalize the segmentation

        image_file = modality_map[modality]
        out_file = get_output_filename (output_patient_dir, modality)
        correct_bias(image_file, out_file)

    seg_file = modality_map[Modality.seg]
    truth_file = get_output_filename(output_patient_dir, Modality.seg)
    shutil.copy(seg_file, truth_file)


