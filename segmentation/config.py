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

config = configparser.ConfigParser()
config.read(default_config_file)
