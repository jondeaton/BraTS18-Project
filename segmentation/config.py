#!/usr/bin/env python
"""
File: config
Date: 5/13/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import configparser

dir_name = os.path.dirname(__file__)
default_config_file = os.path.join(dir_name, "train_config.ini")


class Configuration(object):

    def __init__(self, config_file=default_config_file):
        assert isinstance(config_file, str)
        self._config_file = config_file
        self._config = configparser.ConfigParser()
        self._config.read(self._config_file)
        c = self._config

        self.brats_directory = os.path.expanduser(c["BraTS"]["root"])
        self.tfrecords_dir = os.path.expanduser(c["BraTS"]["TFRecords"])
        self.tensorboard_dir = os.path.expanduser(c["TensorFlow"]["tensorboard-dir"])
        self.model_file = os.path.expanduser(c["Output"]["save-file"])

        self.learning_rate = float(c["Hyperparameters"]["learning-rate"])
        self.learning_decay_rate = float(c["Hyperparameters"]["learning-decay-rate"])
        self.num_epochs = int(c["Hyperparameters"]["epochs"])
        self.mini_batch_size = int(c["Hyperparameters"]["mini-batch"])
        self.seed = int(c["Hyperparameters"]["seed"])

        self.shuffle_buffer_size = int(c["dataset"]["prefetch-buffer-size"])
        self.prefetch_buffer_size = int(c["dataset"]["shuffle-buffer-size"])

    def overload(self, args):
        assert args is not None

        if args.brats_directory is not None:
            self.brats_directory = args.brats_directory

        if args.model_file is not None:
            self.model_file = args.model_file

        if args.tensorboard is not None:
            self.tensorboard_dir = args.tensorboard
