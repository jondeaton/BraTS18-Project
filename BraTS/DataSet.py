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
import nibabel as nib

from BraTS.Patient import *
from BraTS.load_utils import *

_brats_root_dir = None

def set_root(new_brats_root):
    """
    Set the root directory containing multiple BraTS datasets

    :param new_brats_root: The new path to the root directory
    :return: None
    """
    if not isinstance(new_brats_root, str):
        raise Exception("New root was not a string")
    global _brats_root_dir
    _brats_root_dir = new_brats_root

survival_df_cache = {}  # Prevents loading CSVs more than once

class DataSubSet:

    def __init__(self, dir_map, survival_csv):
        self._dir_map = dir_map
        self._patient_ids = list(dir_map.keys())
        self._survival_csv = survival_csv
        self._num_patients = len(self._patient_ids)

        # Data caches
        self._mris = None
        self._segs = None
        self._patients = None
        self._survival_df_cached = None

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
        mris_shape = (self._num_patients,) + mri_shape
        segs_shape = (self._num_patients,) + img_shape

        self._mris = np.empty(shape=mris_shape)
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
                mri, seg = load_patient_data(patient_dir)
                self._mris[i] = mri
                self._segs[i] = seg

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

    @property
    def _survival_df(self):
        if self._survival_df_cached is None:
            if self._survival_csv not in survival_df_cache:
                df = self._load_survival(self._survival_csv)
                survival_df_cache[self._survival_csv] = df
            else:
                df = survival_df_cache[self._survival_csv]
            self._survival_df_cached = df.loc[df.id.isin(self._patient_ids)]
        return self._survival_df_cached

    @classmethod
    def _load_survival(cls, survival_csv):
        try:
            survival = pd.read_csv(survival_csv)
        except:
            raise Exception("Error reading survival CSV file: %s" % survival_csv)
        return cls._rename_columns(survival)

    @classmethod
    def _rename_columns(cls, df):
        # Rename the columns
        if df.shape[1] == 3:
            df.columns = ['id', 'age', 'survival']
        elif df.shape[1] == 4:
            df.columns = ['id', 'age', 'survival', 'resection']
        else:
            raise Exception("Unknown columns in survival: %s" % df.columns)

class DataSet(object):

    def __init__(self, root=None, year=None):

        if root is not None:
            self._root = root

        elif year is not None:
            if _brats_root_dir is None:
                raise Exception("Must set_root before using year argument")

            year_dir = find_file_containing(_brats_root_dir, str(year % 100))
            self._root = os.path.join(_brats_root_dir, year_dir)
        else:
            raise Exception("Pass root or year optional argument")

        self._validation = None
        self._train = None
        self._HGG = False
        self._LGG = False

        self._dir_map_cache = None

        self._val_dir = None
        self._train_dir_cached = None
        self._hgg_dir = os.path.join(self._train_dir, "HGG")
        self._lgg_dir = os.path.join(self._train_dir, "LGG")

        self._train_survival_csv_cached = None
        self._validation_survival_csv_cached = None

        self._train_ids = None
        self._hgg_ids_cached = None
        self._lgg_ids_cached = None

        self._train_dir_map_cache = None
        self._validation_dir_map_cache = None
        self._hgg_dir_map_cache = None
        self._lgg_dir_map_cache = None

    @property
    def train(self):
        """
        Training data

        Loads the training data from disk, utilizing caching
        :return: A tf.data.Dataset object containing the training data
        """
        if self._train is None:
            self._train = DataSubSet(self._train_dir_map, self._train_survival_csv)
        return self._train

    @property
    def validation(self):
        if self._validation is None:
            self._validation = DataSubSet(self._validation_dir_map, self._validation_survival_csv)
        return self._validation

    @property
    def HGG(self):
        if self._HGG is None:
            self._HGG = DataSubSet(self._hgg_dir_map, self._train_survival_csv)
        return self._HGG

    @property
    def LGG(self):
        if self._LGG is None:
            self._HGG = DataSubSet(self._lgg_dir_map, self._train_survival_csv)
        return self._LGG

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
    def _train_survival_csv(self):
        if self._train_survival_csv_cached is None:
            self._train_survival_csv_cached = find_file_containing(self._train_dir, "survival")
            if self._train_survival_csv_cached is None:
                raise Exception("Could not find survival CSV in %s" % self._train_dir)
        return self._train_survival_csv_cached

    @property
    def _validation_survival_csv(self):
        if self._validation_survival_csv_cached is None:
            self._validation_survival_csv_cached = find_file_containing(self._validation_dir, "survival")
            if self._validation_survival_csv_cached is None:
                raise Exception("Could not find survival CSV in %s" % self._validation_dir)
        return self._validation_survival_csv_cached

    @property
    def _train_dir(self):
        if self._train_dir_cached is not None:
            return self._train_dir_cached
        self._train_dir_cached = find_file_containing(self._root, "training")
        if self._train_dir_cached is None:
            raise Exception("Could not find training directory in %s" % self._root)
        return self._train_dir_cached

    @property
    def _validation_dir(self):
        if self._val_dir is not None:
            return self._val_dir
        self._val_dir = find_file_containing(self._root, "validation")
        if self._val_dir is None:
            raise Exception("Could not find validation directory in %s" % self._root)
        return self._val_dir

    @property
    def _train_dir_map(self):
        if self._train_dir_map_cache is None:
            self._train_dir_map_cache = dict(self._hgg_dir_map)
            self._train_dir_map_cache.update(self._lgg_dir_map)
        return self._train_dir_map_cache

    @property
    def _validation_dir_map(self):
        if self._validation_dir_map_cache is None:
            self._validation_dir_map_cache = self._directory_map(self._validation_dir)
        return self._validation_dir_map_cache

    @property
    def _hgg_dir_map(self):
        if self._hgg_dir_map_cache is None:
            self._hgg_dir_map_cache = self._directory_map(self._hgg_dir)
        return self._hgg_dir_map_cache

    @property
    def _lgg_dir_map(self):
        if self._lgg_dir_map_cache is None:
            self._lgg_dir_map_cache = self._directory_map(self._lgg_dir)
        return self._lgg_dir_map_cache

    @property
    def _hgg_ids(self):
        if self._hgg_ids_cached is None:
            self._hgg_ids_cached = os.listdir(self._hgg_dir)
        return self._hgg_ids_cached

    @property
    def _lgg_ids(self):
        if self._lgg_ids_cached is None:
            self._lgg_ids_cached = os.listdir(self._lgg_dir)
        return self._lgg_ids_cached

    @classmethod
    def _directory_map(cls, dir):
        return { file: os.path.join(dir, file)
                 for file in os.listdir(dir)
                 if os.path.isdir(os.path.join(dir, file)) }
