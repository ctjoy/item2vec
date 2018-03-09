# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import string
import jieba
from collections import Counter, OrderedDict

class ItemNameProcessor(object):

    def __init__(self, data, name_col):
        self.data_name = data[name_col]
        self.data_cut = self.cut_name(data[name_col])
        # pd.concat([self.data_name, self.data_cut], axis=1).to_csv('./data/cut.csv', index=False)
        # self.data_cut = data['cut'].fillna('')
        self.word_dict = self.get_word_dict()

        self.word_list = list(self.word_dict.keys())
        self.word_counts = list(self.word_dict.values())

        self.clean_data = self.map_to_ix()

    def cut_name(self, name):

        def rm_short(row):
            return '::::'.join(list(set(filter(lambda x: len(x) > 1, row))))

        start = time.time()
        # 維基百科標點符號 常用的標點符號 中華民國教育部
        punctuation_ch = set(u'，。？！、；：「」『』（）［］〔〕【】—…－-～‧《》〈〉﹏＿')
        exclude = set(string.punctuation) | set(punctuation_ch)

        name = name.apply(lambda x: ''.join(filter(lambda n: n not in exclude, x)))
        name = name.apply(lambda x: list(jieba.cut(x, cut_all=False)))
        name = name.apply(lambda x: rm_short(x))
        name = name.apply(lambda x: '::::'.join(list(tuple(set(x.lower().split('::::'))))))
        print('Cut names (jieba) in {0:.2f} sec'.format(time.time() - start))

        name.fillna('', inplace=True)

        return name

    def get_word_dict(self):
        total_words = self.data_cut.str.cat(sep='::::').split('::::')
        counter = Counter(total_words).most_common(len(total_words))

        return OrderedDict(sorted(filter(lambda v: v[1] > 5, counter), reverse=True, key=lambda v: v[1]))

    def map_to_ix(self):
        word_to_ix = dict(zip(self.word_list, range(len(self.word_list))))
        self.word_to_ix = word_to_ix
        map_to_ix = lambda x: [word_to_ix[i] for i in x.split('::::') if i in self.word_list]

        return self.data_cut.apply(map_to_ix)

    def get_factors(self, embeddings):
        data = self.clean_data
        embedding_size = embeddings.shape[1]

        factors = []

        for i in data.tolist():
            if len(i) == 0:
                factors.append(list(np.full(embedding_size, 0)))
            else:
                factors.append(list(np.mean(embeddings[i], axis=0)))

        self.factors = np.array(factors)

    def get_norms(self):
        self.norms = np.linalg.norm(self.factors, axis=-1)
        self.norms[self.norms == 0] = 1e-10

    def similar_items(self, itemid, N):
        scores = self.factors.dot(self.factors[itemid]) / self.norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / self.norms[itemid]), key=lambda x: -x[1])

    def print_similar_items(self, embeddings, itemid, N=10):
        self.get_factors(embeddings)
        self.get_norms()

        for i, score in self.similar_items(itemid, N):
            print(self.data_name.iloc[i], score)
        print('-'*10)
