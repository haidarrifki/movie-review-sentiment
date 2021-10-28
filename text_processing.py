import pymongo
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

connection_url="mongodb://localhost:27017/" #MongoDB compass local host URL. You can replace the SRV string if you are connecting with mongodb atlas
# connection_url = "mongodb+srv://ngadimin:uvIVS1HWYm6C9MVX@cluster0.sdb0e.mongodb.net/?retryWrites=true&w=majority"
client = pymongo.MongoClient(connection_url)
client.list_database_names() #listing the available databases

db_name = "skripsi"
skripsi_db = client[db_name]
datasets_collection = skripsi_db['datasets']
datasets_backup_collection = skripsi_db['datasets_backup']
text_processings_collection = skripsi_db['text_processings']
classifications_collection = skripsi_db['classifications']
positive_words_collection = skripsi_db['positive_words']
negative_words_collection = skripsi_db['negative_words']
testing_collection = skripsi_db['testing']

stop = set(stopwords.words('english')) 
sno = nltk.stem.SnowballStemmer('english') 

def cleanhtml(sentence): 
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, ' ', sentence)
  return cleantext
def cleanpunc(sentence): 
  cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
  cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
  return cleaned

i=0
str1=' '
final_string=[]
all_positive_words=[]
all_negative_words=[]
s=''
labels=[]
text_processings = text_processings_collection.find({})
for text in text_processings:
  filtered_sentence=[]
  id=text['_id']
  label=text['label']
  text=cleanhtml(text['text'])
  for w in text.split():
    for cleaned_words in cleanpunc(w).split():
      if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
        if(cleaned_words.lower() not in stop):
          s=(sno.stem(cleaned_words.lower())).encode('utf8')
          filtered_sentence.append(s)
          if label == 'pos':
            # insert to positive words collection
            all_positive_words.append(s)
            # positive_words_collection.insert_one({'word': s.decode()})
          if label == 'neg':
            # insert to negative words collection
            all_negative_words.append(s)
            # negative_words_collection.insert_one({'word': s.decode()})
        else:
          continue
      else:
        continue

  # append label
  labels.append(label)
  # join word
  str1 = b" ".join(filtered_sentence)
  # print(str1.decode())
  # text_processings_collection.update_one({ '_id': id },{'$set':{'after': str1.decode()}})
  # if label == 'unsup':
    # insert to classification
    # classifications_collection.insert_one({ 'text': s.decode() })
  final_string.append(str1)
  i+=1

# data['cleaned_review']=final_string

def posneg(x):
  if x=="neg":
    return 0
  elif x=="pos":
    return 1
  return x

print(labels)
filtered_score = labels.map(posneg)
data['score'] = filtered_score
# print(data['score'])

print('ok')