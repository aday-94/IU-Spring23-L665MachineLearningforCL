import spacy
import json
from spacy.tokenizer import Tokenizer
import re
import string
from wordsegment import segment
import pandas as pd

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)
nlp.add_pipe("emoji", first=True)

# regular expression to match user mentions
user_mention_regex = r'@[a-zA-Z0-9_]+'

# spanish punctuations
spanish_punctuation = '¡¿«»'

# add a space after each punctuation mark, except when it is a #
english_punctuation = string.punctuation.replace('#', '')
space_after_punctuation_regex = '[' + english_punctuation + spanish_punctuation + ']'

# punctuations to remove
multilingual_punctuation = string.punctuation + spanish_punctuation
translator = str.maketrans('', '', multilingual_punctuation)


def preprocess_tweet(tweet):
    # convert to lowercase
    tweet = tweet.lower()

    # removing user mentions
    tweet = re.sub(user_mention_regex, '', tweet)

    # removing URLs
    tweet = re.sub(r'http\S+', '', tweet)

    tweet = re.sub(r'([' + re.escape(string.punctuation.replace('#', '') + spanish_punctuation) + '])(?<!#)', r'\1 ', tweet)

    # emojis to their alias
    doc = nlp(tweet)

    segment_next_token = False

    for token in doc:
        if segment_next_token:
            segmented_text = " ".join(segment(token.text))
            tweet = tweet.replace(token.text, segmented_text)
            segment_next_token = False

        if token.text.startswith('#'):
            segment_next_token = True

        if token._.is_emoji:
            tweet = tweet.replace(token.text, token._.emoji_desc)

    # removing punctuations
    tweet = tweet.translate(translator)

    # remove extra spaces between words
    tweet = re.sub('\s+', ' ', tweet)

    return tweet


def preprocess_tweets_file(gold_file_path, data_file_path):
    # load gold label file
    with open(gold_file_path, 'r') as f:
        gold_labels = json.load(f)

    # extract tweet ids from gold label file
    tweet_ids = [id_exist for id_exist in gold_labels]

    # load tweets from file
    with open(data_file_path, 'r') as f:
        tweets_dict = json.load(f)

    # extract tweet text for filtered tweet ids
    tweets = [tweets_dict[id_exist]['tweet'] for id_exist in tweet_ids]

    # preprocess tweets
    preprocessed_tweets = [preprocess_tweet(tweet) for tweet in tweets]

    # create DataFrame with preprocessed tweets
    df_preprocessed = pd.DataFrame({'text': preprocessed_tweets})

    return df_preprocessed