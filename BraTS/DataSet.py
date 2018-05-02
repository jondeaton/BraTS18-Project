#!/usr/bin/env python
"""
File: DataSet
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)

This file provides loading of the BraTS datasets
for ease of use in TensorFlow models.
"""

import os

import pandas as pd
import numpy as np
import tensorflow as tf

import nibabel as nib
from nibabel.testing import data_path

from BraTS.Patient import *
from BraTS.load_utils import *

_brats_root = None
def set_root(new_brats_root):
    _brats_root = new_brats_root


class DataSubSet:

    def __init__(self, survival_df, patient_ids, dir_map):
        self._dir_map = dir_map
        self._patient_ids = list(patient_ids)
        self._survival_df = survival_df
        self._num_patients = len(self._patient_ids)

        # Data caches
        self._mris = None
        self._segs = None
        self._dataset = None
        self._patients = None

    @property
    def mris(self):
        if self._mris is not None:
            return self._mris

        self._load_images()
        return self._mris

    @property
    def segs(self):
        if self._segs is not None:
            return self._segs
        self._load_images()
        return self._segs

    def _load_images(self):
        imgs_shape = (self._num_patients,) + mri_shape
        segs_shape = (self._num_patients,) + imgs_shape

        self._mris = np.empty(shape=imgs_shape)
        self._segs = np.empty(shape=segs_shape)

        # If patients was already loaded
        if self._patients is not None:
            for i, patient in enumerate(self._patients.values()):
                self._mris[i] = patient.mri_data
                self._segs[i] = patient.seg
        else:
            # Load it from scratch
            for i, patient_id in enumerate(self._patient_ids):
                patient_dir = self._dir_map[patient_id]
                self._mris[i], self._segs[i] = load_patient_data(patient_dir)

    @property
    def dataset(self):
        if self._dataset is not None:
            return self._dataset
        return tf.data.Dataset.from_tensor_slices((self.mris, self.segs))

    @property
    def patients(self):
        if self._patients is not None:
            return self._patients

        # Construct the patients dictionary
        self._patients = {}
        for i, patient_id in enumerate(self._patient_ids):
            patient = Patient(patient_id)
            patient.age = self._survival_df.loc[self._survival_df.id == patient_id].Age
            patient.survival = self._survival_df.loc[self._survival_df.id == patient_id].Survival

            if self._mris is not None:
                # Load from _mris and _segs if possible
                patient.mri = self._mris[i]
                patient.seg = self._segs[i]
            else:
                # Load the image and segmentation data
                patient_dir = self._dir_map[patient_id]
                patient.mri, patient.survival = load_patient_data(patient_dir)

            # Store the patient object in the dictionary
            self._patients[patient_id] = patient
        return self._patients


class DataSet(object):

    def __init__(self, root=None, year=None):

        if root is not None:
            self._root = root

        elif year is not None:
            if _brats_root is None:
                raise Exception("Must set_root before using year argument")

            year_dir = find_file_containing(_brats_root, str(year % 100))
            self._root = os.path.join(_brats_root, year_dir)

        else:
            raise Exception("Pass root or year optional argument")

        self._survival = None

        self._train = None
        self._dev = False
        self._validation = False

        self._HGG = False
        self._LGG = False

        self._dir_map_cache = None

        self._train_dir = None
        self._val_dir = None

        self._hgg_dir = os.path.join(self._training_dir, "HGG")
        self._lgg_dir = os.path.join(self._training_dir, "LGG")

        self._train_ids = None
        self._dev_ids = None

    @property
    def train(self):
        """
        Training data

        Loads the training data from disk, utilizing caching
        :return: A tf.data.Dataset object containing the training data
        """
        if self._train is not None:
            return self._train  # return cached value

        ids = os.path.listdir(self._hgg_dir) + os.path.listdir(self._lgg_dir)
        self._train = DataSubSet(self.survival, ids, self._dir_map)
        return self._train

    @property
    def dev(self):
        if self._dev is not None:
            return self._dev
        # todo: split it with train
        ids = os.path.listdir(self._hgg_dir) + os.path.listdir(self._lgg_dir)
        self._dev = DataSubSet(self.survival, ids, self._dir_map)
        return self._dev

    @property
    def validation(self):
        if self._validation is not None:
            return self._validation



    @property
    def HGG(self):
        if self._HGG is not None:
            return self._HGG

        ids = os.path.listdir(self._hgg_dir)
        self._HGG = DataSubSet(self.survival, ids, self._dir_map)
        return self._HGG

    @property
    def LGG(self):
        if self._LGG is not None:
            return self._LGG

        ids = os.path.listdir(self._hgg_dir)
        self._LGG = DataSubSet(self.survival, ids, self._dir_map)
        return self._LGG

    @property
    def survival(self):
        if self._survival is not None:
            return self._survival

        survival_csv = find_file_containing(self._train_dir, "survival")
        if survival_csv is None:
            raise Exception("Couldn't find survival data.")

        self._survival = pd.read_csv(survival_csv)
        self._survival.columns = ['id', 'age', 'survival']
        return self._survival

    def drop_cache(self):
        """
        Drops the cached values in the object
        :return: None
        """
        self._train = None
        self._dev = None
        self._validation = None
        self._survival_df = None

    @property
    def _training_dir(self):
        if self._train_dir is not None:
            return self._train_dir
        self._train_dir = find_file_containing(self._root, "training")
        if self._train_dir is None:
            raise Exception("Could not find training directory in %s" % self._root)
        return self._train_dir

    @property
    def _validation_dir(self):
        if self._val_dir is not None:
            return self._val_dir
        self._val_dir = find_file_containing(self._root, "validation")
        if self._val_dir is None:
            raise Exception("Could not find validation directory in %s" % self._root)
        return self._val_dir

    @property
    def _dir_map(self):
        if self._dir_map_cache is not None:
            return self._dir_map_cache

        self._dir_map_cache = {}
        for path, dirs, files in os.walk(self._root):
            for file in files:
                if self.survival.id.str.contains(file).any:
                    self._dir_map_cache[file] = os.path.join(path, file)
        return self._dir_map_cache
