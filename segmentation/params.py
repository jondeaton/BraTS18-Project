#!/usr/bin/env python
"""
For loading hyper-parameters

From: cs230-code-examples
"""

import os
import json

dir_name = os.path.dirname(__file__)
default_params_file = os.path.join(dir_name, "params.json")

class Params():
    """
    Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path=default_params_file):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)


    @property
    def adam(self):
        return self.dict["adam"]

    @property
    def learning_rate(self):
        return self.dict["learning_rate"]

    @property
    def learning_decay_rate(self):
        return self.dict["learning_decay_rate"]

    @property
    def epochs(self):
        return self.dict["epochs"]

    @property
    def test_batch_size(self):
        return self.dict["test_batch_size"]

    @property
    def mini_batch_size(self):
        return self.dict["mini_batch_size"]

    @property
    def seed(self):
        return self.dict["seed"]

    @property
    def prefetch_buffer_size(self):
        return self.dict["prefetch_buffer_size"]

    @property
    def shuffle_buffer_size(self):
        return self.dict["shuffle_buffer_size"]

    @property
    def augment(self):
        return self.dict["augment"]

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def save_dict_to_json(d, json_path):
    """
    Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)