from pymongo import MongoClient  # MongoDB driver

from pandas import DataFrame

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

connection_url = "mongodb://localhost:27017/"  # MongoDB compass local host URL. You can replace the SRV string if you are connecting with mongodb atlas
client = MongoClient(connection_url)

db_name = "skripsi"
db = client[db_name]
text_processings_collection = db["text_processings"]
terms_collection = db["terms"]
datasets = text_processings_collection.find({})
datasets = DataFrame(list(datasets), dtype=object)

# using the count vectorizer
count = CountVectorizer()
word_count = count.fit_transform(datasets["textProcessed"])
feature_names = count.get_feature_names()

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count)
df_idf = DataFrame(tfidf_transformer.idf_, index=feature_names, columns=["idf_weights"])
idf = DataFrame(word_count.T.todense(), index=feature_names)
idf_sum = idf.sum(axis=1)
idf_result = DataFrame(idf_sum, columns=["idf"])
# inverse document frequency
tf_idf_vector = tfidf_transformer.transform(word_count)
df_tfidf = DataFrame(tf_idf_vector.T.todense(), index=feature_names)
df_tfidf_sum = df_tfidf.sum(axis=1)
df_tfidf_result = DataFrame(df_tfidf_sum, columns=["tfidf"])

words = df_tfidf_result.index.values
df_tfidf_result.insert(0, column="word", value=words)
df_tfidf_result.insert(1, column="df", value=idf_result)

terms = df_tfidf_result.to_dict("records")
terms_collection.insert_many(terms)

print("ok")
