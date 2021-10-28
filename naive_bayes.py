import pymongo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
import seaborn as sns
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from nltk.corpus import stopwords
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

# DB CONFIG
connection_url="mongodb://localhost:27017/" #MongoDB compass local host URL. You can replace the SRV string if you are connecting with mongodb atlas
# connection_url = "mongodb+srv://ngadimin:uvIVS1HWYm6C9MVX@cluster0.sdb0e.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(connection_url)
db_name = "skripsi"
db = client[db_name]

# Collections
datasets_collection = db['datasets']
text_processings_collection = db['text_processings']
classifications_collection = db['classifications']
testing_collection = db['testing']

# Text Processing
def cleanHtml(sentence):
  cleanr = re.compile('<.*?>')
  cleantxt = re.sub(cleanr, ' ', sentence)
  return cleantxt

def cleanpunc(sentence):
  cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
  cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
  return cleaned

# Stemmer
sno = nltk.stem.SnowballStemmer("english")
stop = set(stopwords.words("english"))

# Variable
all_positive_words = []
all_negative_words = []
final_string = []
final_string_train = []
str1 = ''
payload_frame = { 'review': [], 'sentiment': [] }

# Processing text
text_processings = text_processings_collection.find({})
for string in text_processings:
  # Data 
  id=string['_id']
  label=string['label']
  filtered_sentence = []
  # Removes html tags from every review
  sent = cleanHtml(string['text'])
  for w in sent.split():
    # For every word in a review clean punctions
    for cleanwords in cleanpunc(w).split():
      # if cleaned is alphabet and length og words greater than 2 then proceed
      if ((cleanwords.isalpha()) and len(cleanwords)>2):
        # check weather word is stop word or not
        if cleanwords.lower() not in stop:
          # If word is not stop word then append it to filtered sentence
          s = (sno.stem(cleanwords.lower())).encode('utf-8')
          filtered_sentence.append(s)
          if label.lower() == "pos":
            all_positive_words.append(s)
          if label.lower() == "neg":
            all_negative_words.append(s)
        else:
          continue
      else:
        continue

  # filtered_sentence is list contains all words of a review after preprocessing
  # join every word in a list to get a string format of the review
  str1 = b" ".join(filtered_sentence)

  if label.lower() == 'pos':
    payload_frame['sentiment'].append(1)
  elif label.lower() == 'neg':
    payload_frame['sentiment'].append(0)

  if (label.lower() != 'unsup'):
    payload_frame['review'].append(str1)
    final_string.append(str1)
  else:
    final_string_train.append(str1)

  text_processings_collection.update_one({ '_id': id },{'$set':{'textProcessed': str1.decode()}})

data = pd.DataFrame(data=payload_frame)

freq_pos_words = nltk.FreqDist(all_positive_words)
freq_neg_words = nltk.FreqDist(all_negative_words)

freq_pos_words.most_common(15)
freq_neg_words.most_common(15)

# TF-Idf vector using bi-grams
count_vect_tfidf = TfidfVectorizer(ngram_range = (1, 2))
count_vect_tfidf = count_vect_tfidf.fit(data["review"].values)
tfidf_wrds  = count_vect_tfidf.transform(data["review"].values)

X = tfidf_wrds
Y = data["sentiment"]
x_l, x_test, y_l, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# Naive Bayes
clf = MultinomialNB(alpha = 0.7)
clf.fit(x_l, y_l)
pred = clf.predict(x_test)
acc = accuracy_score(y_test, pred, normalize = True) * float(100)  
train_acc = accuracy_score(y_l, clf.predict(x_l), normalize = True) * float(100)

# Print result
print("acc is on test data:", acc)
print("train accuracy is:", train_acc)
print(classification_report(y_test, pred))

# review = ["This is a worst movie","This is a good movie"]
data['review'] = final_string_train
vectorize = CountVectorizer(vocabulary = count_vect_tfidf.vocabulary_)
polarity = clf.predict(vectorize.transform(data['review'].values))
data['sentiment'] = polarity

def conv_string(text):
  return text.decode()

data['review'] = data['review'].map(conv_string)
# Insert to classifications
classifications_collection.insert_many(data.to_dict('records'))

# f = open('classifier.pickle', 'wb')
# pkl.dump(clf, f)
# f.close()

# f = open('vectorizer.pickle', 'wb')
# pkl.dump(count_vect_tfidf, f)
# f.close()

print('ok')