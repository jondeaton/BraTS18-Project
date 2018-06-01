#!/usr/bin/env python
"""
File: records
Date: 5/8/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os


def get_TFRecord_filename(patient_id):
    """
    Gets the name that should be assigned to a TFRecord of a patient ID

    :param patient_id: The ID of the patient
    :return: File name for the TFRecord file
    """
    return "%s.tfrecord.gzip" % patient_id


def get_id_of_TFRecord(tfrecord_file):
    """
    Gets the ID that corresponds to a TFRecord file


    (assumes that it was named with the functino above this one)
    This function effectively inverts the previous function
    :param tfrecord_file: Path/filename of TFRecord file
    :return: The patient ID for that file
    """
    assert isinstance(tfrecord_file, str)
    file_name = os.path.basename(tfrecord_file)
    base = os.path.splitext(os.path.splitext(file_name)[0])[0]
    return base.strip()
