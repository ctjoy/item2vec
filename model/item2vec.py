# -*- coding: utf-8 -*-
import sys
import argparse
import pandas as pd
import numpy as np
from itertools import combinations
from collections import deque
import ast

import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, default='../atest.csv',
                    help='Training text file.')

parser.add_argument('--epochs', type=int, default=40,
                    help='Number of training epochs.')

parser.add_argument('--embedding_size', type=int, default=200,
                    help='The embedding dimension size.')

parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate', type=float, default=0.2,
                    help='Initial learning rate.')

parser.add_argument('--num_negatives', type=int, default=100,
                    help='Negative samples per training example.')

class BatchGenerator(object):

    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.data = pd.read_csv(FLAGS.train_file, names=['items'])
        self.ix = 0
        self.buffer = deque([])

    def next(self):

        while len(self.buffer) < self.batch_size:
            items = self.data.iloc[self.ix]['items']
            items_list = ast.literal_eval(items)
            self.buffer.extend(combinations(items_list, 2))

            if self.ix == self.data.shape[0] - 1:
                self.ix = 0
            else:
                self.ix += 1

        return [self.buffer.popleft() for _ in range(self.batch_size)]

class Item2Vec(object):

    def __init__(self):
        self.vocab_size = 100
        self.embed_dim = FLAGS.embedding_size
        self.num_negatives = FLAGS.num_negatives
        self.learning_rate = FLAGS.learning_rate

        self._init_graphs()

    def _init_graphs():
        self.embeddings = []
        pass

    def train():
        pass


def main(unused_argv):

    generator = BatchGenerator()

    for i in range(2):
        a = generator.next()
        print(a)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
