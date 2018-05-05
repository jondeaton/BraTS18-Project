#!/usr/bin/env python
"""
File: utils
Date: 5/1/18 
Author: Jon Deaton (jdeaton@stanford.edu)
"""

import os
import pandas as pd


def load_survival(survival_csv):
    """
    Loads a survival CSV file
    :param survival_csv: The path to the CSV file to load
    :return: Pandas DataFrame with the survival information
    """
    try:
        survival = pd.read_csv(survival_csv)
    except:
        raise Exception("Error reading survival CSV file: %s" % survival_csv)
    return rename_columns(survival)


def rename_columns(df):
    """
    Rename the columns of a survival data CSV so that they are consistent
    across different data-sets
    :param df: The raw Pandas DataFrame read from the survival CSV file
    :return: The same DataFrame but with the columns modified
    """
    if df.shape[1] == 3:
        df.columns = ['id', 'age', 'survival']
    elif df.shape[1] == 4:
        df.columns = ['id', 'age', 'survival', 'resection']
    else:
        raise Exception("Unknown columns in survival: %s" % df.columns)
    return df


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
        if keyword in (file if case_sensitive else file.lower()):
            return os.path.join(directory, file)
    return None


def find_file_named(root, name):
    """
    Find a file named something

    :param root: Root directory to search recursively through
    :param name: The name of the file to search for
    :return: Full path to the (first!) file with the specified name found,
    or None if no file was found of that name.
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


