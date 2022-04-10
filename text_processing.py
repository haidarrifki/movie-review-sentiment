# import time

# start_time = time.time()

from pymongo import MongoClient  # MongoDB driver
from pandas import DataFrame

# Natural language tool kits
# import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# required on first install
# download stopwords and nltk punctuation
# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")

# string operations
from string import punctuation
import re

connection_url = "mongodb://localhost:27017/"  # MongoDB compass local host URL. You can replace the SRV string if you are connecting with mongodb atlas
# connection_url = "mongodb+srv://ngadimin:uvIVS1HWYm6C9MVX@cluster0.sdb0e.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_url)

db_name = "skripsi"
db = client[db_name]
datasets_collection = db["datasets"]
text_processings_collection = db["text_processings2"]
# find datasets collection from mongodb
datasets = datasets_collection.find()
df = DataFrame(list(datasets))
# delete _id
del df["_id"]
# ge tlist stopwords n punctuations
sw = stopwords.words("english")

# Function transform text for cleansing process
def transform_text(s):

    # remove html
    html = re.compile(r"<.*?>")
    s = html.sub(r"", s)

    # remove numbers
    s = re.sub(r"\d+", "", s)

    # remove punctuation
    # remove stopwords
    tokens = word_tokenize(s)

    new_string = []
    for w in tokens:
        # remove words with len = 2 AND stopwords
        if len(w) > 2 and w not in sw:
            new_string.append(w)

    s = " ".join(new_string)
    s = s.strip()

    exclude = set(punctuation)
    s = "".join(ch for ch in s if ch not in exclude)

    return s.strip()


# Process Stemming text
lemmatizer = WordNetLemmatizer()  # lemmatizer


def lemmatizer_text(s):
    tokens = word_tokenize(s)

    new_string = []
    for w in tokens:
        lem = lemmatizer.lemmatize(w, pos="v")
        # exclude if lenght of lemma is smaller than 2
        if len(lem) > 2:
            new_string.append(lem)

    s = " ".join(new_string)
    return s.strip()


df["textProcessed"] = df["review"].str.lower()
df["textProcessed"] = df["textProcessed"].apply(transform_text)
df["textProcessed"] = df["textProcessed"].apply(lemmatizer_text)

dataframe = df.to_dict("records")
text_processings_collection.insert_many(dataframe)
print("ok")
# print(">>> Bulk Insert 50k data from panda dataframe without chunk")
# seconds = time.time() - start_time
# print("Time Execution:", time.strftime("%H:%M:%S", time.gmtime(seconds)))
