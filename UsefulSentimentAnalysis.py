import numpy as np 
import pandas as pd 
import re 
import nltk 
import matplotlib.pyplot as plt 
import seaborn as sns 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import operator 

# the github link for the airline reviews 
data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
airline_tweets = pd.read_csv(data_source_url)

tweets = airline_tweets.iloc[:,10].values
sentiments = airline_tweets.iloc[:,1].values
# cleanup based on
# https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk

# set the stopwords
stop_words = set(stopwords.words("english"))

# only have to clean the tweets 

# remove all the crap we don't want
processed_tweets = []
for sentence in range(0, len(tweets)):
    # remove special characters 
    processed_tweet = re.sub(r'\W', ' ', str(tweets[sentence]))

    # remove all single characters 
    processed_tweet = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_tweet)

    # remove single characters from the start 
    processed_tweet = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_tweet)

    # substituting multiple spaces with a single space 
    processed_tweet = re.sub(r'\s+', ' ', processed_tweet, flags=re.I)

    # removing prefixed 'b' for bytes
    processed_tweet = re.sub(r'^b\s+', '', processed_tweet)

    # remove numbers
    processed_tweet = re.sub(r'\d+', '', processed_tweet)

    # convert to lower 
    processed_tweet = processed_tweet.lower()

    processed_tweets.append(processed_tweet)

# tokenize the sentences into word arrays and remove stopwords
filtered_tweets = []
for tweet in processed_tweets:
    filtered_tweet = []
    tokenized_tweet = word_tokenize(tweet) 
    for w in tokenized_tweet:
        if w.lower() not in stop_words:
            filtered_tweet.append(w.lower())
    filtered_tweets.append(filtered_tweet)

# lexicon normalization through stemming 
# DO NOT USE THIS SINCE LEMMATIZING IS BETTER
'''ps = PorterStemmer()
stemmed_tweets = [] 
for tweet in filtered_tweets:
    stemmed_tweet = []
    for w in tweet:
        stemmed_tweet.append(ps.stem(w))
    stemmed_tweets.append(stemmed_tweet)
print("number of stemmed tweets: ", len(stemmed_tweets))
print("first stemmed tweet: ", stemmed_tweets[0])'''


# lemmatizing 
# USE THIS SINCE IT'S BETTER THAN STEMMING
lem = WordNetLemmatizer() 
lemmatized_tweets = []
for tweet in filtered_tweets:
    lemmatized_tweet = []
    for w in tweet:
        lemmatized_tweet.append(lem.lemmatize(w))
    lemmatized_tweets.append(lemmatized_tweet)


# set the train and test datasets 
X_train = lemmatized_tweets[:11000]
y_train = sentiments[:11000]

X_test = lemmatized_tweets[11001:]
y_test = sentiments[11001:]


for t in range(0, len(X_train)):
    X_train[t] = ' '.join(X_train[t])

for t in range(0, len(X_test)):
    X_test[t] = ' '.join(X_test[t])

print("y_test: ", len(y_test))
print("y_train: ", len(y_train))
# sentiment analysis based on
# https://towardsdatascience.com/machine-learning-text-processing-1d5a2d638958
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)
print(train_vectors.shape, test_vectors.shape)

# train using the Multinomial Naive Bayes Classifier
clf = MultinomialNB().fit(train_vectors, y_train)
print("model trained")

predicted = clf.predict(test_vectors)
print(accuracy_score(y_test, predicted))



# Random Forest Classifier
full_features = list(X_train) + list(X_test)
labels = list(y_train) + list(y_test)

vectorizer = TfidfVectorizer(max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
full_features = vectorizer.fit_transform(full_features).toarray()

text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
#X_train2, X_test2, y_train2, y_test2 = train_test_split(full_features, labels, test_size=0.2, random_state=0)
X_train2 = full_features[:11000]
y_train2 = labels[:11000]
X_test2 = full_features[11001:]
y_test2 = labels[11001:]

text_classifier.fit(X_train2, y_train2)
predictions = text_classifier.predict(X_test2)
print(accuracy_score(y_test2, predictions))



positive_words = set()
negative_words = set()

full_words = dict()
for p in range(0, len(predictions)):
    if predictions[p] == "positive":
        for word in X_test[p].split():
            if word in full_words.keys():
                full_words[word] += 1
            else:
                full_words[word] = 1
    if predictions[p] == "negative":
        for word in X_test[p].split():
            if word in full_words.keys():
                full_words[word] -= 1
            else:
                full_words[word] = -1

for k,v in full_words.items():
    if full_words[k] > 0:
        positive_words.add(k)
    if full_words[k] < 0:
        negative_words.add(k)

# positive_words and negative_words are sets of words 
# full_words is a dict with key words and their pos/neg count 
# pos_word_color and neg_word_color are dict of green:list(positive_words) and red:list(negative_words)
pos_word_color = dict()
neg_word_color = dict()


# sort positive and negative words 
most_positive = dict(sorted(full_words.items(), key=operator.itemgetter(1), reverse=True))
pos_list = []
count = 0
for k,v in most_positive.items():
    if count < 20:
        pos_list.append(k)
        count += 1

most_negative = dict(sorted(full_words.items(), key=operator.itemgetter(1), reverse=False))
neg_list = []
count = 0
for k,v in most_negative.items():
    if count < 20:
        neg_list.append(k)
        count += 1

#negative_words.remove("flight")
#positive_words.add("flight")

# this works for sending over all positive and negative words 
pos_word_color["green"] = list(positive_words)
neg_word_color["red"] = list(negative_words)


# this works for sending over only the top 10 positive and negative words 
#pos_word_color["green"] = pos_list
#neg_word_color["red"] = neg_list