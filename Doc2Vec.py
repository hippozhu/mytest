import csv
import time
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from random import shuffle
from random import random
import multiprocessing
import itertools
import pandas as pd

def iter_job_desc(threshold=1.0, n_report=100000):
    with open('jobs5m_clean_deduplicate.csv', 'rt') as fin:
        csvreader = csv.reader(fin)
        next(csvreader)
        t0 = time.time()
        for i, row in enumerate(csvreader):
            if (i+1) % n_report == 0:
                print('%.1fm docs in %.1fs' %((i+1)/1e6, time.time()-t0))
            if random()<threshold:
                yield TaggedDocument(words=row[2].split(), tags=[row[0], row[3]])

def train_model(model ,alpha_delta, passes):
    t0 = time.time()
    for epoch in range(passes):
        #shuffle(jobs)
        model.train(iter_job_desc(0.667, 500000))
        model.alpha -= alpha_delta
        model.min_alpha -= alpha_delta
        print('Pass %d finished in %.1fs' %(epoch, time.time()-t0))
    #return model

def calc_vec(job):
    #model, job = model_job
    return job.tags[0], model.infer_vector(job.words)

def get_doc2vec():
    pool = multiprocessing.Pool(16)
    iter_results = pool.imap(calc_vec, iter_job_desc(), 10000)
    df_doc2vec = pd.DataFrame.from_items(iter_results, columns=range(300), orient='index')
    pool.close()
    pool.join()
    return df_doc2vec

#model = Doc2Vec.load('model.doc2vec')
if __name__ == '__main__':
    #jobs = list(iter_job_desc())
    dimension = 500
    model = Doc2Vec(size=dimension, workers=16)
    model.build_vocab(iter_job_desc())
    alpha, min_alpha, passes = (0.025, 0.001, 20)
    alpha_delta = (alpha - min_alpha) / passes
    model.alpha, model.min_alpha = alpha, alpha
    train_model(model, alpha_delta, passes)
    model.save('model_%d_%d.doc2vec' %(dimension, passes))
