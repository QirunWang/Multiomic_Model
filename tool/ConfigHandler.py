#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/02/11 09:34
# @Author  : Kieran Wang
# @File    : ConfigHandler.py
# @Software: PyCharm

import json
import os

class ConfigHandler:
    """
    This class is used to write out the configuration json files for different other classes.
    """
    def __init__(self, config_dir="./config/"):
        self.config_dir = config_dir

    def dict_to_json(self, config_dict, file_name):
        # Write a dict to a JSON file
        file_path=os.path.join(self.config_dir, file_name)
        with open(file_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)  # `indent=4` makes it pretty-printed

    def json_to_dict(self, file_name):
        # load a json
        file_path = os.path.join(self.config_dir, file_name)
        with open(file_path, "r") as f:
            para_dict = json.load(f)
        return para_dict






