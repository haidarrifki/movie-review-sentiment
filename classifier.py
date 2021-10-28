import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer

review = ["This is a worst movie","This is a good movie"]

with open("classifier.pickle", 'rb') as f:
  classifier = pkl.load(f)
with open("vectorizer.pickle", 'rb') as f:
  vectorizer = pkl.load(f)

#vectorize your review which you used in training
vector_review = CountVectorizer(vocabulary = vectorizer.vocabulary_)
vector_review = vector_review.transform(review)
#predict the vectorized review using your classifier 
predict = classifier.predict(vector_review)
print(predict)