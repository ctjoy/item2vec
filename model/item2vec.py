# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from itertools import combinations
from collections import deque

class Options(object):

    def __init__(self, embedding_size, batch_size,
                 learning_rate, num_negatives, save_path):
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_negatives = num_negatives
        self.save_path = save_path

class BatchGenerator(object):

    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.ix = 0
        self.buffer = deque([])
        self._finish = False

    def next(self):

        if self._finish:
            return 'No data!'

        while len(self.buffer) < self.batch_size:
            items_list = self.data.iloc[self.ix]
            self.buffer.extend(combinations(items_list, 2))

            if self.ix == self.data.shape[0] - 1:
                self._finish = True
                self.ix = 0
            else:
                self.ix += 1

        d = [self.buffer.popleft() for _ in range(self.batch_size)]
        d = np.array([[i[0], i[1]] for i in d])
        batch = d[:, 0]
        labels = d[:, 1]

        return batch , labels

    @property
    def finish(self):
        return self._finish

    def resume(self):
        self.ix = 0
        self._finish = False

    @property
    def current_percentage(self):
        return (self.ix / self.data.shape[0]) * 100

class Item2Vec(object):

    def __init__(self, session, opts, processor):
        self.vocab_size = len(processor.word_list)
        self.embed_dim = opts.embedding_size
        self.num_negatives = opts.num_negatives
        self.learning_rate = opts.learning_rate
        self.batch_size = opts.batch_size
        self.save_path = opts.save_path
        self.step = 0

        # Clean up the model directory if present
        if os.path.exists(self.save_path):
            shutil.rmtree(self.save_path, ignore_errors=False)

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.item_counts = processor.word_counts
        self.generator = BatchGenerator(opts.batch_size,
                                        processor.clean_data)
        self.processor = processor
        meta_path = os.path.join(self.save_path, 'word_metadata.tsv')
        # self.processor.generate_word_meta(meta_path)
        meta_data = self.processor.get_word_meta()
        meta_data.to_csv(meta_path, sep='\t', index=False)

        self.projector_config = projector.ProjectorConfig()
        pro_embed = self.projector_config.embeddings.add()
        pro_embed.tensor_name = 'word_embedding'
        # Link this tensor to its metadata file (e.g. labels).
        pro_embed.metadata_path = 'word_metadata.tsv'

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

        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter(self.save_path,
                                                    graph=tf.get_default_graph())

        projector.visualize_embeddings(self.summary_writer, self.projector_config)

    def forward(self, batch, labels):

        init_width = 0.5 / self.embed_dim
        embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_dim],
                                            -init_width, init_width),
                            name='word_embedding')
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
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        train_op = optimizer.minimize(loss)

        tf.summary.scalar("loss", loss)
        self.summary_op = tf.summary.merge_all()

        return train_op

    @property
    def embeddings(self):
        return self.embed.eval()

    def get_norms(self):
        norms = np.linalg.norm(self.embeddings, axis=-1)
        norms[norms == 0] = 1e-10
        return norms

    def similar_items(self, itemid, N=10):
        norms = self.get_norms()
        scores = self.embeddings.dot(self.embeddings[itemid]) / norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / norms[itemid]), key=lambda x: -x[1])

    def train(self):
        avg_loss = 0
        while not self.generator.finish:
            batch, labels = self.generator.next()
            feed_dict = {self.batch: batch, self.labels: labels}
            _, loss_val, summary = self.session.run([self.train_op, self.loss, self.summary_op],
                                           feed_dict=feed_dict)

            self.summary_writer.add_summary(summary, self.step)

            avg_loss += loss_val
            self.step += 1

        print("Cost: ", '{:.9f}'.format(avg_loss))
        self.generator.resume()
        self.saver.save(self.session,
                        os.path.join(self.save_path, "model.ckpt"),
                        global_step=self.step)

    def evaluate(self, word):
        ix = self.processor.word_to_ix[word]
        for i, score in self.similar_items(ix):
            print(self.processor.word_list[i], score)
        print('-'*10)
