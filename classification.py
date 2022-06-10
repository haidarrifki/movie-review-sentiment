from pymongo import MongoClient  # MongoDB driver
from pandas import DataFrame, concat, Series, crosstab
from nltk import word_tokenize
import numpy as np
import math

connection_url = "mongodb://localhost:27017/"  # MongoDB compass local host URL. You can replace the SRV string if you are connecting with mongodb atlas
# connection_url = "mongodb+srv://ngadimin:uvIVS1HWYm6C9MVX@cluster0.sdb0e.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_url)

db_name = "skripsi"
db = client[db_name]
examinations_collection = db["examinations"]
text_processings_collection = db["text_processings"]
# find datasets collection from mongodb
datasets = text_processings_collection.find()
df = DataFrame(list(datasets))

# Train dataset (first 17.500 rows)
pos_train = df[df["sentiment"] == "positive"][["textProcessed", "sentiment"]].head(2000)
neg_train = df[df["sentiment"] == "negative"][["textProcessed", "sentiment"]].head(2000)

# Test dataset (last 7.500 rows)
pos_test = df[df["sentiment"] == "positive"][["textProcessed", "sentiment"]].tail(500)
neg_test = df[df["sentiment"] == "negative"][["textProcessed", "sentiment"]].tail(500)

# put all toghether again...
train_df = concat([pos_train, neg_train]).sample(frac=1).reset_index(drop=True)
test_df = concat([pos_test, neg_test]).sample(frac=1).reset_index(drop=True)

# print(train_df.head())
# print(test_df.head())


def get_word_counts(words):
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0.0) + 1.0
    return word_counts


def fit(df_fit):
    num_messages = {}
    log_class_priors = {}
    word_counts = {}
    vocab = set()

    n = df_fit.shape[0]
    num_messages["positive"] = df_fit[df_fit["sentiment"] == "positive"].shape[0]
    num_messages["negative"] = df_fit[df_fit["sentiment"] == "negative"].shape[0]
    log_class_priors["positive"] = math.log(num_messages["positive"] / n)
    log_class_priors["negative"] = math.log(num_messages["negative"] / n)
    word_counts["positive"] = {}
    word_counts["negative"] = {}

    for x, y in zip(df_fit["textProcessed"], df_fit["sentiment"]):

        counts = get_word_counts(word_tokenize(x))
        for word, count in counts.items():
            if word not in vocab:
                vocab.add(word)
            if word not in word_counts[y]:
                word_counts[y][word] = 0.0

            word_counts[y][word] += count

    return word_counts, log_class_priors, vocab, num_messages


word_counts, log_class_priors, vocab, num_messages = fit(train_df)
word_count_df = (
    DataFrame(word_counts)
    .fillna(0)
    .sort_values(by="positive", ascending=False)
    .reset_index()
)
# print(word_count_df)


def predict(df_predict, vocab, word_counts, num_messages, log_class_priors):
    result = []
    for x in df_predict:
        counts = get_word_counts(word_tokenize(x))
        positive_score = 0
        negative_score = 0
        for word, _ in counts.items():
            if word not in vocab:
                continue

            # add Laplace smoothing
            log_w_given_positive = math.log(
                (word_counts["positive"].get(word, 0.0) + 1)
                / (num_messages["positive"] + len(vocab))
            )
            log_w_given_negative = math.log(
                (word_counts["negative"].get(word, 0.0) + 1)
                / (num_messages["negative"] + len(vocab))
            )

            positive_score += log_w_given_positive
            negative_score += log_w_given_negative

        positive_score += log_class_priors["positive"]
        negative_score += log_class_priors["negative"]

        if positive_score > negative_score:
            result.append("positive")
        else:
            result.append("negative")
    return result


result = predict(
    test_df["textProcessed"], vocab, word_counts, num_messages, log_class_priors
)
# print(result[0:10])  # result sample...
y_true = test_df["sentiment"].tolist()

acc = sum(1 for i in range(len(y_true)) if result[i] == y_true[i]) / float(len(y_true))
acc_score = "{:.1%}".format(acc)
# print("{0:.4f}".format(acc))
# print("{:.1%}".format(acc))

y_actu = Series(y_true, name="Real")
y_pred = Series(result, name="Predicted")
df_confusion = crosstab(y_actu, y_pred)
df_confusion.astype(np.int32)
# print(df_confusion)
# print(df_confusion["negative"][0])  # true_positive
# print(df_confusion["negative"][1])  # false_negative
# print(df_confusion["positive"][0])  # false_positive
# print(df_confusion["positive"][1])  # true_negative
# df_confusion = df_confusion / df_confusion.sum(axis=1) * 100
# print(df_confusion.round(2))
# print("\n")
# print(df_confusion)

# delete old result first
examinations_collection.delete_many({})

examine_data = {
    "accuracy": acc_score,
    "true_positive": int(df_confusion["negative"][0]),
    "false_negative": int(df_confusion["negative"][1]),
    "false_positive": int(df_confusion["positive"][0]),
    "true_negative": int(df_confusion["positive"][1]),
}

examinations_collection.insert_one(dict(examine_data))

print("ok")
