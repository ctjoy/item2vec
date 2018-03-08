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

parser = argparse.ArgumentParser()

parser.add_argument('--train_file', type=str, default='../test.csv',
                    help='Training text file.')

parser.add_argument('--epochs', type=int, default=1,
                    help='Number of training epochs.')

parser.add_argument('--embedding_size', type=int, default=30,
                    help='The embedding dimension size.')

parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate', type=float, default=0.2,
                    help='Initial learning rate.')

parser.add_argument('--num_negatives', type=int, default=100,
                    help='Negative samples per training example.')

class BatchGenerator(object):

    def __init__(self, batch_size, items, item_ix):
        self.batch_size = batch_size
        self.data = items
        self.ix = 0
        self.item_ix_reverse = {v: i for i, v in enumerate(item_ix)}
        self.buffer = deque([])
        self.finish = False

    def next(self):

        if self.finish:
            return 'No data!'

        while len(self.buffer) < self.batch_size:
            items_list = self.data.iloc[self.ix]
            self.buffer.extend(combinations(items_list, 2))

            if self.ix == self.data.shape[0] - 1:
                self.finish = True
            else:
                self.ix += 1

        d = [self.buffer.popleft() for _ in range(self.batch_size)]
        d = np.array([[self.item_ix_reverse[i[0]], self.item_ix_reverse[i[1]]] for i in d])
        batch = d[:, 0]
        labels = d[:, 1]

        return batch , labels

    def check_finish(self):
        return self.finish

    def resume(self):
        self.ix = 0
        self.finish = False

    def get_current(self):
        return (self.ix / self.data.shape[0]) * 100

class Item2Vec(object):

    def __init__(self, session, items, item_counts, vocab_size, item_ix):
        self.vocab_size = vocab_size
        self.embed_dim = FLAGS.embedding_size
        self.num_negatives = FLAGS.num_negatives
        self.learning_rate = FLAGS.learning_rate
        self.batch_size = FLAGS.batch_size
        # self.num_steps = 10000

        self.item_counts = item_counts
        self.item_ix_reverse = {v: i for i, v in enumerate(item_ix)}
        self.generator = BatchGenerator(FLAGS.batch_size, items, item_ix)

        self.session = session
        self._init_graphs()

    def _init_graphs(self):
        self.batch = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        self.labels = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])
        true_logits, sampled_logits = self.forward(self.batch,
                                                   self.labels)
        self.loss = self.nce_loss(true_logits, sampled_logits)
        # tf.summary.scalar("NCE_loss", self.loss)
        self.train_op = self.optimize(self.loss)

        tf.global_variables_initializer().run()
        # self.saver = tf.train.Saver()

    def forward(self, batch, labels):

        init_width = 0.5 / self.embed_dim
        embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim],
                                            -init_width, init_width))
        self.embed = embed

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
        example_emb = tf.nn.embedding_lookup(embed, batch)

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
        global_step = tf.Variable(0, name="global_step")
        self.global_step = global_step
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss,
                                      global_step=self.global_step)

        return train_op

    def get_factors(self):
        return self.embed.eval()

    def similar_items(self, itemid, N=10):
        item_factors = self.embed.eval()
        item_norms = np.linalg.norm(item_factors, axis=-1)
        item_norms[item_norms == 0] = 1e-10

        scores = item_factors.dot(item_factors[itemid]) / item_norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / item_norms[itemid]), key=lambda x: -x[1])

    def train(self):
        avg_loss = 0
        step = 0
        while not self.generator.check_finish():
            batch, labels = self.generator.next()
            feed_dict = {self.batch: batch, self.labels: labels}
            _, loss_val = self.session.run([self.train_op, self.loss],
                                           feed_dict=feed_dict)
            avg_loss += loss_val
            step += 1
            if step % 1000 == 0:
                print('{:.2f} %'.format(self.generator.get_current()))
                print(loss_val)
                for i, score in self.similar_items(1):
                    print(i, score)
                print('-'*10)

        print(avg_loss)
        self.generator.resume()

def main(unused_argv):

    min_rating = 4
    ratings = pd.read_csv('../data/ml-20m/ratings.csv')
    positive = ratings[ratings.rating >= min_rating]
    items = positive.groupby('userId')['movieId'].apply(list).reset_index(drop=True)
    item_counts = list(positive.movieId.value_counts().sort_index())
    print(len(item_counts))
    movies_ix = list(positive.movieId.unique())
    movies_ix_reverse = {v: i for i, v in enumerate(movies_ix)}
    vocab_size = len(movies_ix)

    with tf.Graph().as_default(), tf.Session() as session:
        with tf.device("/cpu:0"):
            model = Item2Vec(session, items, item_counts, vocab_size, movies_ix)

        for _ in range(FLAGS.epochs):
            model.train() # Process one epoch

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
