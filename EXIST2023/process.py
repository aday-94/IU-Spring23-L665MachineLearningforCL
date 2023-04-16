import spacy
import json
from spacy.tokenizer import Tokenizer
import re

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)
nlp.add_pipe("emoji", first=True)

# regular expression to match user mentions
user_mention_regex = r'@[a-zA-Z0-9_]+'

f = open('EXIST2023_training.json', 'r')
training = json.loads(f.read())

print('working')

for id_exist in training:

    print(training[id_exist]['tweet'])

    # actual tweet
    tweet = training[id_exist]['tweet']

    # removing user mentions
    tweet = re.sub(user_mention_regex, '', tweet)

    # emojis to their alias
    doc = nlp(tweet)

    for token in doc:
        if token._.is_emoji:
            tweet = tweet.replace(token.text, token._.emoji_desc)

    training[id_exist]['tweet'] = tweet

f.close()
