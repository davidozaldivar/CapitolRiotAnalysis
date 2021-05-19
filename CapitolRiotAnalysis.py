"""
 David Zaldivar

 This program:

  1. prints TFIDF 100/500 to csv files,
  2. Calculates ngrams (char and word), printing to csv file
  3. Plots ngrams (char and word), showing graph.
  4. Prints 3 KDE graphs based on density of the compound score, for the target,
        for without the target, and combined.
  5. Prints a violin graph timeline that matches up to the events of the riot.

"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import collections
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import nltk
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import seaborn as sns

sns.set_theme()

stop_words = nltk.corpus.stopwords.words('english')

df = pd.read_csv('Tweets_2021-01-06_CLEANED_Mar12.csv')

# First drop columns I don't need
df = df.drop(['user_name', 'location_name', 'longitude', 'latitude', 'user_location'], axis=1)

print("dropping duplicates")
df = df.drop_duplicates(subset=['tweet_id'])
print("done dropping duplicates..")

# and drop text column with missing data
df = df.dropna(subset=['text'])


# Find if text contains 'capitol'
def findCapitol(x):
    if 'capitol' in x:
        return 1
    else:
        return 0


print("Making Capitol...")

df['capitol'] = df.apply(lambda x: findCapitol(str(x['text'])), axis=1)

# Separating df for graph, and want retweets counted
df_forGraph = df.groupby('capitol').apply(lambda x: x.sample(n=1749, random_state=19)).reset_index(drop=True)

# drop duplicates (retweets)
print("dropping retweets (duplicates)..")
df = df.drop_duplicates(subset=['text'])
print("done dropping retweets..")

# Now that we have capitol, use group by based on it
train_data = df.groupby('capitol').apply(lambda x: x.sample(n=410)).reset_index(drop=True)

'''

  TFIDF 

'''


def createTFDF(rows, features):
    name = "sample_" + str(rows)
    halfRows = int(rows / 2)

    # 50% should have Capitol keyword.
    df_sample_2000 = train_data.groupby('capitol').apply(lambda x: x.sample(n=halfRows)).reset_index(drop=True)

    print(df_sample_2000.head(10))

    print("Calculating top term frequency by inverse document frequency matrix...")
    vectorizer = CountVectorizer(max_df=0.95, max_features=features, binary=False)
    counts = vectorizer.fit_transform(df_sample_2000.text)

    # Transforms the data into a bag of words
    # count_train = vectorizer.fit(df_sample_2000.title)
    bag_of_words = vectorizer.transform(df_sample_2000.text)
    # find maximum value for each of the features over dataset:
    max_value = bag_of_words.max(axis=0).toarray().ravel()
    sorted_by_tfidf = max_value.argsort()
    # get feature names
    feature_names = np.array(vectorizer.get_feature_names())
    print("Features with lowest tfidf:\n{}".format(
        feature_names[sorted_by_tfidf[:features]]))
    print("\nFeatures with highest tfidf: \n{}".format(
        feature_names[sorted_by_tfidf[-features:]]))
    # find maximum value for each of the features over all of dataset:
    max_val = bag_of_words.max(axis=0).toarray().ravel()
    # sort weights from smallest to biggest and extract their indices
    sort_by_tfidf = max_val.argsort()
    print("Features with lowest tfidf:\n{}".format(
        feature_names[sort_by_tfidf[:features]]))
    print("\nFeatures with highest tfidf: \n{}".format(
        feature_names[sort_by_tfidf[-features:]]))
    TDFCount_Vectorizer = pd.DataFrame(counts.toarray(), columns=vectorizer.get_feature_names())
    name1 = name + "TFIDF" + str(features) + ".csv"
    TDFCount_Vectorizer.to_csv(name1)
    print(TDFCount_Vectorizer)
    dfnew = pd.read_csv(name1)

    combinedDF = pd.concat([df_sample_2000, dfnew], ignore_index=False, sort=False, axis=1, join="inner")
    name2 = "combined" + name1
    combinedDF.to_csv(name2)
    text = " ".join(df_sample_2000.text)
    # Create the wordcloud object
    wordcloud = WordCloud(width=1024, height=1024, margin=0).generate(text)

    # Display the generated image:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    ax.margins(x=0, y=0)
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.set_title("Target Status")
    explode = (0, 0.1)
    labels = '0', '1'
    ax.pie(list(dict(collections.Counter(list(df_sample_2000.capitol))).values()), explode=explode, labels=labels,
           autopct='%1.1f%%', shadow=True, startangle=90)


print("Making 100/500 TFIDF...")
# get TFDF : using smaller sample size for brevity and the need to remove
# retweet duplicates.
createTFDF(820, 100)
createTFDF(820, 500)
print("Done making 100/500 TFIDF.")

'''


#  NGRAM Section

'''

import re
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter


def cleanReviews(documents):
    cleanedReviews = []
    for document in documents:
        s = re.sub(r'[^a-zA-Z0-9\s]', '', document)
        s = re.sub('\s+', ' ', s)
        s = str(s).lower()
        tokens = [token for token in s.split(" ") if token != ""]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = [word for word in tokens if
                  word not in ['amp', 'us', 'united states', 'im', 'call', 'january', '6th', 'dont', 'december', 'dc',
                               '19',
                               'capitol', 'united', 'states', 'gif', 'section', 'keyword', 'time', 'go', 'gt', 'um']]
        review = ' '.join(tokens)
        cleanedReviews.append(review)
    return (cleanedReviews)


def documentNgramsWords(documents, size):
    ngrams_all = []
    for document in documents:
        tokens = document.split()
        if len(tokens) <= size:
            continue
        else:
            output = list(ngrams(tokens, size))
        for ngram in output:
            ngrams_all.append(" ".join(ngram))
    cnt_ngram = Counter()
    for word in ngrams_all:
        cnt_ngram[word] += 1
    df = pd.DataFrame.from_dict(cnt_ngram, orient='index').reset_index()
    df = df.rename(columns={'index': 'words', 0: 'count'})
    df = df.sort_values(by='count', ascending=False)
    df = df.head(10)
    df = df.sort_values(by='count')
    return (df)


def splittoChar(word):
    return [char.strip() for char in word]


def documentNgramsChars(documents, size):
    ngrams_all = []

    for document in documents:
        tokens = splittoChar(document)
        if len(tokens) <= size:
            continue
        else:
            output = list(ngrams(tokens, size))
        for ngram in output:
            # Added: Ensures appended values are chars == to size
            if len("".join(ngram)) == size:
                ngrams_all.append("".join(ngram))
    cnt_ngram = Counter()
    for word in ngrams_all:
        cnt_ngram[word] += 1
    df = pd.DataFrame.from_dict(cnt_ngram, orient='index').reset_index()
    df = df.rename(columns={'index': 'words', 0: 'count'})
    df = df.sort_values(by='count', ascending=False)
    df = df.head(10)
    df = df.sort_values(by='count')
    return (df)


# only interested in tweets containing 'capitol'
query_hasCap = train_data[train_data['capitol'] == 1]

# Clean and separate both sets of texts
txt_hasCapCleaned = cleanReviews(query_hasCap['text'])
text_all = cleanReviews(train_data['text'])

# get ngrams for target (capitol), using capitol set of texts
uni_word = documentNgramsWords(txt_hasCapCleaned, 1)
bi_word = documentNgramsWords(txt_hasCapCleaned, 2)
tri_word = documentNgramsWords(txt_hasCapCleaned, 3)

uni_char = documentNgramsChars(txt_hasCapCleaned, 1)
bi_char = documentNgramsChars(txt_hasCapCleaned, 2)
tri_char = documentNgramsChars(txt_hasCapCleaned, 3)

# make lists of word values
uniList_word = uni_word['words'].values.tolist()
biList_word = bi_word['words'].values.tolist()
triList_word = tri_word['words'].values.tolist()

uniList_char = uni_char['words'].values.tolist()
biList_char = bi_char['words'].values.tolist()
triList_char = tri_char['words'].values.tolist()

# make columns based on ngram words, fill with 0
for col in uniList_word:
    train_data[col] = 0

for col in biList_word:
    train_data[col] = 0

for col in triList_word:
    train_data[col] = 0

for col in uniList_char:
    train_data[col] = 0

for col in biList_char:
    train_data[col] = 0

for col in triList_char:
    train_data[col] = 0


# This function will find if the column str appears in both the text
# and the list of ngram word values.  If it does, the cell is changed to '1'
def findsIfInText(u, myList, df, index_List):
    global index
    index = index + 1

    rowText = str(u)

    for col in df.columns:
        strCol = str(col)
        if (strCol in rowText) and (strCol in myList):
            df.loc[index_List[index], col] = '1'


# get index list, will be used to iterate over
index_List = list(train_data.index.values.tolist())

# global index
index = -1
train_data.text.apply(lambda u: findsIfInText(u, uniList_word, train_data, index_List))

index = -1
train_data.text.apply(lambda u: findsIfInText(u, biList_word, train_data, index_List))

index = -1
train_data.text.apply(lambda u: findsIfInText(u, triList_word, train_data, index_List))

index = -1
train_data.text.apply(lambda u: findsIfInText(u, uniList_char, train_data, index_List))

index = -1
train_data.text.apply(lambda u: findsIfInText(u, biList_char, train_data, index_List))

index = -1
train_data.text.apply(lambda u: findsIfInText(u, triList_char, train_data, index_List))

print("Printing CSV!")
train_data.to_csv("DZ_MS2_nGrams.csv")

'''

 #   plotting nGram graph


'''


def plotNgrams(documents, subTitle):
    unigrams = documentNgramsWords(documents, 1)
    bigrams = documentNgramsWords(documents, 2)
    trigrams = documentNgramsWords(documents, 3)
    quadgram = documentNgramsWords(documents, 4)

    ## char

    unigrams_char = documentNgramsChars(documents, 1)
    bigrams_char = documentNgramsChars(documents, 2)
    trigrams_char = documentNgramsChars(documents, 3)
    quadgram_char = documentNgramsChars(documents, 4)

    # Set plot figure size
    fig = plt.figure(figsize=(30, 10))
    plt.subplots_adjust(wspace=.5)
    plt.suptitle(subTitle)

    ## top row , word ngrams

    ax = fig.add_subplot(241)
    ax.barh(np.arange(len(unigrams['words'])), unigrams['count'], align='center', alpha=.5)
    ax.set_title('Unigrams')
    plt.yticks(np.arange(len(unigrams['words'])), unigrams['words'])
    plt.xlabel('Count')

    ax2 = fig.add_subplot(242)
    ax2.barh(np.arange(len(bigrams['words'])), bigrams['count'], align='center', alpha=.5)
    ax2.set_title('Bigrams')
    plt.yticks(np.arange(len(bigrams['words'])), bigrams['words'])
    plt.xlabel('Count')

    ax3 = fig.add_subplot(243)
    ax3.barh(np.arange(len(trigrams['words'])), trigrams['count'], align='center', alpha=.5)
    ax3.set_title('Trigrams')
    plt.yticks(np.arange(len(trigrams['words'])), trigrams['words'])
    plt.xlabel('Count')

    ax4 = fig.add_subplot(244)
    ax4.barh(np.arange(len(quadgram['words'])), quadgram['count'], align='center', alpha=.5)
    ax4.set_title('Quad')
    plt.yticks(np.arange(len(quadgram['words'])), quadgram['words'])
    plt.xlabel('Count')

    #### bottom row , char grams

    ax6 = fig.add_subplot(246)
    ax6.barh(np.arange(len(bigrams_char['words'])), bigrams_char['count'], align='center', alpha=.5)
    ax6.set_title('Bigrams')
    plt.yticks(np.arange(len(bigrams_char['words'])), bigrams_char['words'])
    plt.xlabel('Count')

    ax7 = fig.add_subplot(247)
    ax7.barh(np.arange(len(trigrams_char['words'])), trigrams_char['count'], align='center', alpha=.5)
    ax7.set_title('Trigrams')
    plt.yticks(np.arange(len(trigrams_char['words'])), trigrams_char['words'])
    plt.xlabel('Count')

    ax8 = fig.add_subplot(248)
    ax8.barh(np.arange(len(quadgram_char['words'])), quadgram_char['count'], align='center', alpha=.5)
    ax8.set_title('Quad')
    plt.yticks(np.arange(len(quadgram_char['words'])), quadgram_char['words'])
    plt.xlabel('Count')

    # cleans up layout
    fig.tight_layout()
    plt.savefig("TargetHasCapitol_Plot.png")
    plt.show()


def textTrends(documents, subTitle):
    cleanedReviews = cleanReviews(documents)
    plotNgrams(cleanedReviews, subTitle)


print("Getting nGrams graph (for target) ready..")

# only ngrams of tweets with target ('capitol' keyword).
hasCap = train_data[train_data['capitol'] == 1]

textTrends(hasCap['text'], "Target - Has Capitol")

print("Done plotting nGrams.")

'''

For Seaborn graphs: This section produces kde graphs based on the compound score obtained 
    from sentiment analysis.  It computes graphs for: 
    1. Tweets only with the target ('capitol') keyword
    2. Tweets without the target
    3. All tweets (those with and without the target).

    It also produces a violin graph based on the compound score, that is plotted alongside
    the hour obtained from the tweetID.  This makes a timeline that can be compared to
    the timeline of the capitol riot: 

    https://en.wikipedia.org/wiki/Timeline_of_the_2021_storming_of_the_United_States_Capitol


'''

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
df_forGraph['Vader Sentiment Analysis'] = df_forGraph['text'].apply(
    lambda x: analyzer.polarity_scores(str(x)))

df_forGraph['% neg'] = df_forGraph['text'].apply(
    lambda x: round((analyzer.polarity_scores(str(x))['neg'] * 100), 2))

df_forGraph['% neu'] = df_forGraph['text'].apply(
    lambda x: round((analyzer.polarity_scores(str(x))['neu'] * 100), 2))

df_forGraph['% pos'] = df_forGraph['text'].apply(
    lambda x: round((analyzer.polarity_scores(str(x))['pos'] * 100), 2))

df_forGraph['compound'] = df_forGraph['text'].apply(
    lambda x: round((analyzer.polarity_scores(str(x))['compound'] * 100), 2))


# note: neutral has been removed, and range of compound changed from 0.05
def decideSentiment(x):
    if x['compound'] >= 0.01:
        return 'positive'
    elif x['compound'] <= -0.01:
        return 'negative'
    # else:
    # return 'neutral'


df_forGraph['overall sentiment'] = df_forGraph['text'].apply(
    lambda x: decideSentiment(analyzer.polarity_scores(str(x))))

# drop any that are empty for overall sentiment
df_forGraph = df_forGraph.dropna(subset=['overall sentiment'])


# this may be used for graphing purposes later
def get_tweet_timestamp(tid):
    offset = 1288834974657
    tstamp = (tid >> 22) + offset
    utcdttime_orig = datetime.utcfromtimestamp(tstamp / 1000)
    utcdttime_washington = utcdttime_orig + timedelta(hours=-5)

    UTC_datetime_timestamp = int(utcdttime_washington.strftime("%s"))
    UTC_datetime_converted = datetime.fromtimestamp(UTC_datetime_timestamp)

    return int(UTC_datetime_converted.strftime("%H%M%S"))


# this is used to get the hour, for plotting the timeline
def get_tweet_hour(tid):
    offset = 1288834974657
    tstamp = (tid >> 22) + offset
    utcdttime_orig = datetime.utcfromtimestamp(tstamp / 1000)
    utcdttime_washington = utcdttime_orig + timedelta(hours=-5)

    UTC_datetime_timestamp = int(utcdttime_washington.strftime("%s"))
    UTC_datetime_converted = datetime.fromtimestamp(UTC_datetime_timestamp)

    return int(UTC_datetime_converted.strftime("%H"))


# make time columns
df_forGraph["time"] = df_forGraph.tweet_id.apply(lambda u: get_tweet_timestamp(u))
df_forGraph["hour"] = df_forGraph.tweet_id.apply(lambda u: get_tweet_hour(u))

'''
     graph! 

'''

# print 3 kde graphs
fig, axes = plt.subplots(1, 3)
sns.kdeplot(data=df_forGraph[df_forGraph['capitol'] == 1], x="compound", hue="overall sentiment", multiple="stack",
            ax=axes[0])
sns.kdeplot(data=df_forGraph[df_forGraph['capitol'] == 0], x="compound", hue="overall sentiment", multiple="stack",
            ax=axes[1])
sns.kdeplot(data=df_forGraph, x="compound", hue="overall sentiment", multiple="stack", ax=axes[2])
axes[0].set_title("With Target")
axes[1].set_title("Without Target")
axes[2].set_title("Everything")

print("Saving DZ_MS2_KDE.png")
plt.savefig("DZ_MS2_KDE.png", dpi=300)
plt.show()

# violin graph
sns.catplot(data=df_forGraph, kind="violin", x="hour", y="compound", hue="overall sentiment", split=True)
plt.title("Capitol Riot Violin plot timeline (all data)")

print("Saving DZ_MS2_Violin.png")
plt.savefig("DZ_MS2_Violin.png", dpi=300)
plt.show()





