#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/1/22 14:31
# @Author  : Luni Hu
# @File    : log_manager.py
# @Software: PyCharm

import logging

def singleton(cls):
    instances = {}

    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return getinstance

@singleton
class LogManager:
    def __init__(self, log_path=None, log_level=logging.INFO, log_name="multiminer"):
        self.logger = logging.getLogger(log_name)
        self.level = log_level
        self.logger.setLevel(log_level)
        self.formatter = logging.Formatter(
            "%(asctime)s-%(module)s[line-%(lineno)s]-%(levelname)s: %(message)s",
            datefmt='%Y-%m-%d %H:%M:%S')
        self.file_handler = logging.FileHandler(log_path) if log_path else None
        self.stream_handler = logging.StreamHandler()
        self.log_path = log_path
        self.add_handler()

    def add_handler(self):
        if self.file_handler is not None:
            self.file_handler.setFormatter(self.formatter)
            self.file_handler.setLevel(self.level)
            self.logger.addHandler(self.file_handler)
        self.stream_handler.setFormatter(self.formatter)
        self.stream_handler.setLevel(self.level)
        self.logger.addHandler(self.stream_handler)
