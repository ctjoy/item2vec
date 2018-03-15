# -*- coding: utf-8 -*-
import time
import os
import shutil
import argparse
import logging
import numpy as np
import pandas as pd

from model.item2vec import Item2Vec, Options
from utils.process import ItemNameProcessor

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

logging.basicConfig(level=logging.INFO)
logging.info('Start ... ')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='Where did you put your data?',
                    default='./data/cut.csv')
                    # required=True)

parser.add_argument('--save_path', type=str, help='The path you want to save your model.',
                    default='./result/model/')

parser.add_argument('--save_project_path', type=str, help='The path you want to save your projected factors.',
                    default='./result/project/')

parser.add_argument('--epochs', type=int, default=50,
                    help='Number of training epochs.')

parser.add_argument('--embedding_size', type=int, default=30,
                    help='The embedding dimension size.')

parser.add_argument('--batch_size', type=int, default=256,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate', type=float, default=0.5,
                    help='Initial learning rate.')

parser.add_argument('--num_negatives', type=int, default=100,
                    help='Negative samples per training example.')


args = parser.parse_args()


start_time = time.time()
data = pd.read_csv(args.data)

processor = ItemNameProcessor(data, name_col='name')
opts = Options(args.embedding_size,
               args.batch_size,
               args.learning_rate,
               args.num_negatives,
               args.save_path)

with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):
        model = Item2Vec(session, opts, processor)

    for epoch in range(args.epochs):
        model.train() # Process one epoch
        model.evaluate('手機')

        if (epoch + 1) % 5 == 0:
            embeds = model.embeddings
            processor.print_similar_items(embeds, 9, N=10)

        print('Finish {} epoch!'.format(epoch + 1))
        print('-'*10)

    embeds = model.embeddings
    processor.print_similar_items(embeds, 4839, N=10)
    processor.print_similar_items(embeds, 500, N=10)

with tf.Graph().as_default(), tf.Session() as session:
    with tf.device("/cpu:0"):

        # Clean up the model directory if present
        if os.path.exists(args.save_project_path):
            shutil.rmtree(args.save_project_path, ignore_errors=False)

        if not os.path.exists(args.save_project_path):
            os.makedirs(args.save_project_path)

        f = processor.get_factors(embeds)
        f_tf = tf.convert_to_tensor(f, np.float64)

        factors = tf.Variable(f_tf, name="item_embedding")

        meta_path = os.path.join(args.save_project_path, 'item_metadata.tsv')
        processor.generate_item_meta(meta_path)

        saver = tf.train.Saver([factors])

        session.run(factors.initializer)
        saver.save(session,
                   os.path.join(args.save_project_path, "factors.ckpt"))

        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = 'item_embedding'
        # Link this tensor to its metadata file (e.g. labels).
        embedding.metadata_path = 'item_metadata.tsv'
        projector.visualize_embeddings(tf.summary.FileWriter(args.save_project_path), config)

logging.info('Complete in {} minutes!'.format((time.time() - start_time) / 60))
