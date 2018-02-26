# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd

def read_data(path, min_rating=4.0):
    """ Reads in the dataset, and filters down ratings down to positive only"""
    ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
    positive = ratings[ratings.rating >= min_rating]

    movies = pd.read_csv(os.path.join(path, "movies.csv"))

    items = positive.groupby('userId')['movieId'].apply(list)

    return ratings, movies, items

def calculate_similar_movies(input_path, min_rating=4.0):
    # read in the input data file
    logging.debug("reading data from %s", input_path)
    start = time.time()
    ratings, movies, items  = read_data(input_path, min_rating=min_rating)
    print(ratings.head())

    logging.debug("read data file in %s", time.time() - start)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates related movies from the MovieLens 20M "
                                     "dataset (https://grouplens.org/datasets/movielens/20m/)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input', type=str, default='../data/ml-20m/',
                        dest='inputfile', help='Path of the unzipped ml-20m dataset', required=True)

    parser.add_argument('--min_rating', type=float, default=4.0, dest='min_rating',
                        help='Minimum rating to assume that a rating is positive')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)

    calculate_similar_movies(args.inputfile, min_rating=args.min_rating)
