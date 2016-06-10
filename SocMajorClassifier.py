import pandas as pd
import numpy as np
import itertools
import gzip
import re
import ast
import csv
import pickle
import multiprocessing
import scipy
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

import gensim
from gensim.parsing.preprocessing import STOPWORDS

token_pattern = re.compile(r"(?u)\b\w\w+\b")
from SocMajorUtility import stemmer

def iter_jobs():
    #discard_dids = set(line.strip() for line in open('discard_dids'))
    discard_dids = set([])
    with open('jobs5m_deduplicate.csv', 'rt') as fin:
        csvreader = csv.reader(fin)
        next(csvreader)
        for i, row in enumerate(csvreader):
            if row[0] not in discard_dids:
                yield row[2]
            if (i+1)%1000==0:
                print((i+1)/1000000, 'm, ', end='')

class JobCorpus():
    def __init__(self, input_file, title=False, dictionary=None):
        self.input_file = input_file
        self.dictionary = dictionary
        self.csv_idx = 1 if title else 2

    def __iter__(self):
        t0 = time.time()
        with open(self.input_file, 'rt') as fin:
            csvreader = csv.reader(fin)
            next(csvreader)
            for i, row in enumerate(csvreader):
                if self.dictionary:
                    yield self.dictionary.doc2bow(my_tokenizer(row[self.csv_idx]))
                else:
                    yield my_tokenizer(row[self.csv_idx])
                if (i+1)%10000==0:
                    ndocs_processed = i+1
                    ndocs_remained = self.len()-i-1
                    time_elapsed = time.time()-t0
                    time_remained = time_elapsed/ndocs_processed*ndocs_remained
                    print('%.2fm, %dmin, %.1fmin'
                        %(ndocs_processed/1000000, time_elapsed/60, time_remained/60))

    def __len__(self):
        #return self.dictionary.num_docs
        return 1153973

    def len(self):
        #return self.dictionary.num_docs
        return 1153973

def my_tokenizer(s):
    return token_pattern.findall(s.lower())

def my_tokenizer_with_stemming(s):
    return list(map(stemmer.stem, token_pattern.findall(s.lower())))

def my_text_clean(s):
    return ' '.join(my_tokenizer(s))
def convert_bow(row_vectorizer):
    bow, vectorizer = row_vectorizer
    return vectorizer.transform([bow])

def transform_desc(vectorizer):
    job_iterator = iter_jobs()
    pool = multiprocessing.Pool(16)
    iter_results = pool.imap(convert_bow, zip(job_iterator, itertools.repeat(vectorizer)), 20000)
    results = list(iter_results)
    pool.close()
    pool.join()
    return scipy.sparse.vstack(results)

def load_data_clean():
    df_data = pd.read_csv('jobs5m_clean_deduplicate.csv', usecols=['JobDID', 'Title', 'Label'], index_col=0).fillna('')
    df_data_test = pd.read_csv('jobs1500_deduplicate.csv', usecols=['JobDID', 'Title', 'Desc', 'SOC'], encoding="ISO-8859-1", index_col=0)
    df_data_test['SOCList'] = df_data_test.SOC.apply(lambda s: ast.literal_eval(s.replace('OR', ',')))
    df_data_test['Desc'] = df_data_test.Desc.map(my_cleaner)
    df_data_test['Title'] = df_data_test.Title.map(my_cleaner)
    return df_data, df_data_test

def load_data():
    #with open('jobs5m_deduplicate.csv', 'rt') as fin:
    df_data = pd.read_csv('jobs5m_deduplicate.csv', usecols=['JobDID', 'Title', 'Label'], index_col=0, dtype={'Label':np.int8}).fillna('')
    df_data_test = pd.read_csv('jobs1500_deduplicate.csv', usecols=['JobDID', 'Title', 'Desc', 'SOC'], encoding="ISO-8859-1", index_col=0)
    df_data_test['SOCList'] = df_data_test.SOC.apply(lambda s: ast.literal_eval(s.replace('OR', ',')))
    return df_data, df_data_test

def deunigram(X, vectorizer):
    X_lil = X.tolil()
    terms = vectorizer.get_feature_names()
    idx_bigram = set(v for k, v in vectorizer.vocabulary_.items() if ' ' in k)
    for i in range(X.shape[0]):
        idx_term = X[i].nonzero()[1]
        idx_unigram = [vectorizer.vocabulary_[t] for t in set(itertools.chain.from_iterable([terms[it].split() for it in idx_term if it in idx_bigram]))]
        X_lil[i, idx_unigram] = 0
        if (i+1)%100000==0:
            print(i+1)
    return X_lil.tocsr()

def pipe_line(df_data, id_train, id_val, id_test, vectorizer):
    X_title_train = vectorizer.fit_transform(df_data.loc[id_train].Title)
    X_title_val = vectorizer.transform(df_data.loc[id_val].Title)
    X_title_test = vectorizer.transform(df_data.loc[id_test].Title)

if __name__ == '__main__':
    '''
    df_data, df_data_test = load_data()
    vectorizer = CountVectorizer(binary=True, min_df=5, ngram_range=(1, 3), stop_words='english')
    X_title = vectorizer.fit_transform(df_data.Title)
    X_title_test = vectorizer.transform(df_data_test.Title)
    X_desc_test = vectorizer.transform(df_data_test.Desc)
    X_desc = transform_desc(vectorizer)
    '''
    id2word = gensim.corpora.Dictionary.load('id2word.save');stop_ids = [id2word.token2id[stopword] for stopword in STOPWORDS if stopword in id2word.token2id];uncommon_ids = [tokenid for tokenid, docfreq in id2word.dfs.items() if docfreq<3];id2word.filter_tokens(stop_ids + uncommon_ids);id2word.compactify()
    jc = JobCorpus('jobs5m_deduplicate.csv', id2word)
    lda = gensim.models.ldamulticore.LdaMulticore(jc, id2word=id2word, num_topics=500, chunksize=1000, passes=1, workers=6)
    lda.save('lda.save')
