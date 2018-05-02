#!/usr/bin/env python
"""
File: utils
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os


def find_file_containing(directory, keyword, case_sensitive=False):
    """
    Finds a file in a directory containing a keyword in it's name

    :param directory: The directory to search in
    :param keyword: The keyword to search in the name for
    :param case_sensitive: Search with case sensitivity
    :return: The joined path to the file containing the keyword in
    it's name, if found, else None.
    """

    if not isinstance(directory, str):
        raise ValueError("root_dir is not a string.")

    if not isinstance(keyword, str):
        raise ValueError("keyword is not a string")

    if not os.path.isdir(directory):
        raise FileNotFoundError("Not found: %s" % directory)

    # Iterate through files
    for file in os.listdir(directory):
        if keyword in file if case_sensitive else file.lower():
            return os.path.join(directory, file)
    return None


def find_file_named(root, name):
    """
    Finds the directory containing the imaging data for a given patient ID
    :param brats_root: Root directory to the
    :return:
    """

    if not isinstance(root, str):
        raise ValueError("root is not a string.")

    if not isinstance(name, str):
        raise ValueError("name is not a string")

    # Search the directory recursively
    for path, dirs, files in os.walk(root):
        for file in files:
            if file == name:
                return os.path.join(path, file)
    return None


