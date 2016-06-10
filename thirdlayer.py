import pandas as pd
from SocMajorClassifier import my_tokenizer_with_stemming
from sklearn.neighbors import KNeighborsClassifier

def my_pred(soc_X):
    soc, X = soc_X
    knn = KNeighborsClassifier(n_neighbors=1, metric='cityblock').fit(X_soc[socs==soc], df_soc.Minor.loc[socs==soc])
    return knn.predict(X)[0]

df_soc = pd.read_csv('soc_structure_2010.csv', usecols=['Minor','Title']).fillna(method='ffill').drop_duplicates(['Minor','Title']).dropna(axis=0)
minor = df_soc.drop_duplicates('Minor').set_index('Minor')

df_taxonomy = pd.read_csv('df_taxonomy.csv').dropna(subset=['Title'])
df_groups = df_taxonomy.groupby('GroupID').Title.apply(lambda x: '. '.join(x))
glabel = df_taxonomy.drop_duplicates(['GroupID']).set_index('GroupID')['GLabel']

vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words='english', tokenizer=my_tokenizer_with_stemming, binary=True)
X_soc = vectorizer.fit_transform(df_soc.Title)
X_group = vectorizer.transform(df_groups)
pool = multiprocessing.Pool(16)
results = pool.map(my_pred, zip(socs_g, X_group))
pool.close();pool.join()

df_3rd_layer = pd.concat([pd.Series(results, index=df_groups.index), glabel], axis=1)
df_3rd_layer.merge(minor, how='left', left_on='nn1_local', right_index=True).to_csv('ThirdLayer.csv', index_label='GroupID')

