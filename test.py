# Source: https://www.kaggle.com/ganesh562/naive-bayes-on-imdb-and-deploying-naive-bayes/notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, classification_report
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import re
from nltk.corpus import stopwords
import nltk
from collections import Counter
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
import pickle as pkl

data = pd.read_csv('imdb_dataset.csv')
print(data.columns)
print(data["sentiment"].value_counts())

# Text Processing
def cleanHtml(sentence):
  cleanr = re.compile('<.*?>')
  cleantxt = re.sub(cleanr, ' ', sentence)
  return cleantxt

def cleanpunc(sentence):
  cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
  cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
  return cleaned

sno = nltk.stem.SnowballStemmer("english")
stop = set(stopwords.words("english"))
all_positive_words = []
all_negative_words = []
final_string = []
str1 = ''
i = 0
for string in data["review"].values:
  filtered_sentence = []
  # Removes html tags from every review
  sent = cleanHtml(string)
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
          if (data["sentiment"].values)[i].lower() == "positive":
            all_positive_words.append(s)
          if (data["sentiment"].values)[i].lower() == "negative":
            all_negative_words.append(s)
        else:
          continue
      else:
        continue
  # filtered_sentence is list contains all words of a review after preprocessing
  # join every word in a list to get a string format of the review
  str1 = b" ".join(filtered_sentence)
  #append all the string(cleaned reviews)to final_string
  final_string.append(str1)
  i += 1

print(len(all_positive_words))
print(len(all_negative_words))

data["review"] = final_string

def conv_label(label):
  if label.lower() == "positive":
    return 1
  elif label.lower() == "negative":
    return 0

print(data['sentiment'])
data["sentiment"] = data["sentiment"].map(conv_label)

# print(data.head(10))
print(data)
exit()

freq_pos_words = nltk.FreqDist(all_positive_words)
freq_neg_words = nltk.FreqDist(all_negative_words)

freq_pos_words.most_common(15)
freq_neg_words.most_common(15)

#Bag of words vector with bi-grams
count_vect = CountVectorizer(ngram_range = (1, 2))
count_vect = count_vect.fit(data["review"].values)
bigram_wrds = count_vect.transform(data["review"].values)

#TF-Idf vector using bi-grams
count_vect_tfidf = TfidfVectorizer(ngram_range = (1, 2))
count_vect_tfidf = count_vect_tfidf.fit(data["review"].values)
tfidf_wrds  = count_vect_tfidf.transform(data["review"].values)

print(bigram_wrds)

# change X to bigram_wrds to run classifier on Bag Of Words(BoW)
X = bigram_wrds
# X = tfidf_wrds
Y = data["sentiment"]
x_l, x_test, y_l, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

clf = MultinomialNB(alpha = 0.7)
clf.fit(x_l, y_l)
pred = clf.predict(x_test)
acc = accuracy_score(y_test, pred, normalize = True) * float(100)  
print("acc is on test data:", acc)
# sns.heatmap(confusion_matrix(y_test, pred), annot = True, fmt = 'd')
train_acc = accuracy_score(y_l, clf.predict(x_l), normalize = True) * float(100)
print("train accuracy is:", train_acc)
print(classification_report(y_test, pred))

review = ["This is a worst movie","This is a good movie"]
#initialize BOW vectorizer
#we already fitted the model for train data on "count_vect"(means alredy found probabilities for train data)
vectorize = CountVectorizer(vocabulary = count_vect.vocabulary_)
#Use classifier we trained using Bag of words
polarity = clf.predict(vectorize.transform(review))
# count_vect_tfidf.transform(review)
print(polarity)

# f = open('classifier.pickle', 'wb')
# pkl.dump(clf, f)
# f.close()

# f = open('vectorizer.pickle', 'wb')
# pkl.dump(count_vect, f)
# f.close()