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

# The root directory of all the BraTS data-sets
_brats_root_dir = None


def set_root(new_brats_root):
    """
    Set the root directory containing multiple BraTS datasets

    :param new_brats_root: The new path to the root directory
    :return: None
    """
    if not isinstance(new_brats_root, str):
        raise TypeError("Expected root to be a string")
    global _brats_root_dir
    _brats_root_dir = new_brats_root


survival_df_cache = {}  # Prevents loading CSVs more than once


class DataSubSet:

    def __init__(self, dir_map, survival_csv):
        self._dir_map = dir_map
        self._patient_ids = sorted(list(dir_map.keys()))
        self._survival_csv = survival_csv
        self._num_patients = len(self._patient_ids)

        # Data caches
        self._mris = None
        self._segs = None
        self._patients = {}
        self._survival_df_cached = None
        self._patients_fully_loaded = False
        self._id_indexer = {patient_id: i for i, patient_id in enumerate(self._patient_ids)}

    def subset(self, patient_ids):
        """
        Split this data subset into a small subset by patient ID

        :param n: The number of elements in the smaller training set
        :return: A new data subset with only the specified number of items
        """
        dir_map = {id: self._dir_map[id] for id in patient_ids}
        return DataSubSet(dir_map, self._survival_csv)

    @property
    def ids(self):
        return self._patient_ids

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


        if self._patients_fully_loaded:
            # All the patients were already loaded
            for i, patient in enumerate(self._patients.values()):
                self._mris[i] = patient.mri_data
                self._segs[i] = patient.seg
        else:
            # Load it from scratch
            for i, patient_id in enumerate(self._patient_ids):
                patient_dir = self._dir_map[patient_id]
                load_patient_data(patient_dir, mri_array=self._mris, seg_array=self._segs, index=i)

    @property
    def patients(self):
        """
        Loads ALL of the patients from disk into patient objects

        :return: A dictionary containing ALL patients
        """

        if not self._patients_fully_loaded:
            # Construct the patients dictionary
            self._patients = {}
            for i, patient_id in enumerate(self._patient_ids):
                self._patients[patient_id] = self.patient(patient_id)
            self._patients_fully_loaded = True
        return self._patients

    def patient(self, patient_id):
        """
        Loads only a single patient from disk

        :param patient_id: The patient ID
        :return: A Patient object loaded from disk
        """

        if patient_id not in self._patient_ids:
            raise ValueError("Patient id \"%s\" not present." % patient_id)

        if patient_id in self._patients:
            # Return cached value if present
            return self._patients[patient_id]

        patient = Patient(patient_id)

        df = self._survival_df
        if patient_id in df.id.values:
            patient.age = float(df.loc[df.id == patient_id].age)
            patient.survival = int(df.loc[df.id == patient_id].survival)

        if self._mris is not None and self._segs is not None:
            # Load from _mris and _segs if possible
            index = self._id_indexer[patient_id]
            patient.mri = self._mris[index]
            patient.seg = self._segs[index]
        else:
            # Load the mri and segmentation data from disk
            patient_dir = self._dir_map[patient_id]
            patient.mri, patient.seg = load_patient_data(patient_dir)

        self._patients[patient_id] = patient  # cache the value for later
        return patient

    @property
    def _survival_df(self):
        if self._survival_csv in survival_df_cache:
            return survival_df_cache[self._survival_csv]
        df = load_survival(self._survival_csv)
        survival_df_cache[self._survival_csv] = df
        return df



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
        self._hgg = False
        self._lgg = False

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
        """
        Validation data

        :return: Validation data
        """
        if self._validation is None:
            self._validation = DataSubSet(self._validation_dir_map, self._validation_survival_csv)
        return self._validation

    @property
    def hgg(self):
        if self._hgg is None:
            self._hgg = DataSubSet(self._hgg_dir_map, self._train_survival_csv)
        return self._hgg

    @property
    def lgg(self):
        if self._lgg is None:
            self._lgg = DataSubSet(self._lgg_dir_map, self._train_survival_csv)
        return self._lgg

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
