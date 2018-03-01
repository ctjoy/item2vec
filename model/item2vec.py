# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import sys
import argparse
import pandas as pd
from itertools import combinations
from collections import deque
import ast

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, default='../test.csv',
                    help='Training text file.')

parser.add_argument('--epochs', type=int, default=1,
                    help='Number of training epochs.')

parser.add_argument('--embedding_size', type=int, default=30,
                    help='The embedding dimension size.')

parser.add_argument('--batch_size', type=int, default=16,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate', type=float, default=0.2,
                    help='Initial learning rate.')

parser.add_argument('--num_negatives', type=int, default=100,
                    help='Negative samples per training example.')

class BatchGenerator(object):

    def __init__(self, batch_size):
        self.batch_size = batch_size
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

        d = [self.buffer.popleft() for _ in range(self.batch_size)]
        d = np.array([list(i) for i in d])
        batch = d[:, 0]
        labels = d[:, 1]

        return batch , labels

class Item2Vec(object):

    def __init__(self, session, item_counts, vocab_size):
        self.vocab_size = vocab_size
        self.embed_dim = FLAGS.embedding_size
        self.num_negatives = FLAGS.num_negatives
        self.learning_rate = FLAGS.learning_rate
        self.batch_size = FLAGS.batch_size
        self.num_steps = 100

        self.item_counts = item_counts
        self.generator = BatchGenerator(FLAGS.batch_size)

        self.session = session
        self._init_graphs()

    def _init_graphs(self):
        self.batch = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        true_logits, sampled_logits = self.forward(self.batch,
                                                   self.labels)
        self.loss = self.nce_loss(true_logits, sampled_logits)
        self.train_op = self.optimize(self.loss)

        tf.global_variables_initializer().run()

    def forward(self, batch, labels):

        init_width = 0.5 / self.embed_dim
        emb = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim],
                                            -init_width, init_width))
        self.emb = emb

        softmax_w = tf.Variable(tf.zeros([self.vocab_size, self.embed_dim]),
                                name="softmax_weights")
        softmax_b = tf.Variable(tf.zeros([self.vocab_size]),
                                 name="softmax_bias")

        labels_matrix = tf.reshape(tf.cast(labels, dtype=tf.int64),
                            [self.batch_size, 1])

        # Negative sampling
        sampled_ids, _, _ = tf.nn.fixed_unigram_candidate_sampler(
                true_classes=labels_matrix,
                num_true=1,
                num_sampled=self.num_negatives,
                unique=True,
                range_max=self.vocab_size,
                distortion=0.75,
                unigrams=self.item_counts)

        # Embeddings for examples: [batch_size, embed_dim]
        example_emb = tf.nn.embedding_lookup(emb, batch)

        # Weights for labels: [batch_size, embed_dim]
        true_w = tf.nn.embedding_lookup(softmax_w, labels)
        # Biases for labels: [batch_size, 1]
        true_b = tf.nn.embedding_lookup(softmax_b, labels)

        # Weights for sampled ids: [batch_size, embed_dim]
        sampled_w = tf.nn.embedding_lookup(softmax_w, sampled_ids)
        # Biases for sampled ids: [batch_size, 1]
        sampled_b = tf.nn.embedding_lookup(softmax_b, sampled_ids)

        # True logits: [batch_size, 1]
        true_logits = tf.reduce_sum(tf.multiply(example_emb, true_w), 1) + true_b

        # Sampled logits: [batch_size, num_sampled]
        sampled_b_vec = tf.reshape(sampled_b, [self.num_negatives])
        sampled_logits = tf.matmul(example_emb,
                                   sampled_w,
                                   transpose_b=True) + sampled_b_vec

        return true_logits, sampled_logits

    def nce_loss(self, true_logits, sampled_logits):
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(true_logits), logits=true_logits)

        sampled_xent = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(sampled_logits), logits=sampled_logits)

        nce_loss_tensor = (tf.reduce_sum(true_xent) +
                           tf.reduce_sum(sampled_xent)) / self.batch_size

        return nce_loss_tensor

    def optimize(self, loss):
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)

        return train_op

    def train(self):
        for step in range(self.num_steps):
            batch, labels = self.generator.next()
            feed_dict = {self.batch: batch, self.labels: labels}
            _, loss_val = self.session.run([self.train_op, self.loss],
                                           feed_dict=feed_dict)

def main(unused_argv):

    ratings = pd.read_csv('../data/ml-20m/ratings.csv')
    item_counts = list(ratings.movieId.value_counts().sort_index())
    print(len(item_counts))
    vocab_size = int(ratings.movieId.max())

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = Item2Vec(session, item_counts, vocab_size)

        for _ in range(FLAGS.epochs):
            model.train() # Process one epoch

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
