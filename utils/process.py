# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import string
import jieba
from collections import Counter, OrderedDict

class ItemNameProcessor(object):

    def __init__(self, data, name_col):
        self.data = data
        self.data_name = data[name_col]
        self.data_cut = self.cut_name(data[name_col])
        self.word_dict = self.get_word_dict()

        self.word_list = list(self.word_dict.keys())
        self.word_counts = list(self.word_dict.values())

        self.clean_data = self.map_to_ix()

    def get_word_meta(self):
        return pd.DataFrame(self.word_list, columns=['word']).reset_index()

    def get_item_meta(self):
        return self.data.reset_index()

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

        return OrderedDict(sorted(filter(lambda v: v[1] > 5 and v[0] != '', counter), reverse=True, key=lambda v: v[1]))

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

        return np.array(factors)

    def get_norms(self, e):
        norms = np.linalg.norm(e, axis=-1)
        norms[norms == 0] = 1e-10
        return norms

    def get_similar(self, embeddings, queryid, norms, N):
        scores = embeddings.dot(embeddings[queryid]) / norms
        best = np.argpartition(scores, -N)[-N:]
        return sorted(zip(best, scores[best] / norms[queryid]), key=lambda x: -x[1])

    def print_similar(self, embeddings, queyid, N=10, is_item=True):

        if is_item:
            e = self.get_factors(embeddings)
        else:
            e = embeddings
        norms = self.get_norms(e)

        for i, score in self.get_similar(e, queyid, norms, N):
            if is_item:
                print(self.data_name.iloc[i], score)
            else:
                print(self.word_list[i], score)
        print('-'*10)
