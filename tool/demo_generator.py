#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/03/19 09:42
# @Author  : Kieran Wang
# @File    : demo_generator.py
# @Software: PyCharm

import random

class demo_input:
    def __init__(self, n_batch, n_len, m_len):
        """
        Initializes the generator with:
        :param n_batch: number of elements in the minibatch
        :param n_len: length for gene-related lists (gID, gExpr, gPos)
        :param m_len: length for atac-related lists (aPos, aExpr)
        """
        self.n_batch = n_batch
        self.n_len = n_len
        self.m_len = m_len

    def generate(self):
        """
        Generates a minibatch which is a list of dictionaries.
        Each dictionary contains:
            - gID: list of n_len unique random integers from 0 to 1000.
            - gExpr: list of n_len random floats between 0 and 10.
            - gPos: list of n_len random integers between 1 and 1e6.
            - aPos: list of m_len random integers between 1 and 1e6.
            - aExpr: list of m_len random floats between 0 and 10.
            - Metadata: a dictionary with fixed values {'A': 'aaa', 'B': 'bbb'}.
        :return: list of dictionaries (minibatch)
        """
        minibatch = []
        for _ in range(self.n_batch):
            element = {
                'gID': random.sample(range(0, 1001), self.n_len),
                'gExpr': [random.uniform(0, 10) for _ in range(self.n_len)],
                'gPos': [random.randint(1, int(1e6)) for _ in range(self.n_len)],
                'aPos': [random.randint(1, int(1e6)) for _ in range(self.m_len)],
                'aExpr': [random.uniform(0, 10) for _ in range(self.m_len)],
                'Metadata': {'A': 'aaa', 'B': 'bbb'}
            }
            minibatch.append(element)
        return minibatch

# Example usage:
if __name__ == "__main__":
    # Create an instance of demo_generator with:
    # 5 elements in the minibatch, each gene-related list having 10 items,
    # and each atac-related list having 8 items.
    generator = demo_input(n_batch=5, n_len=1000, m_len=10000)
    minibatch = generator.generate()
    print(minibatch)













