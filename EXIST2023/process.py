import spacy
import json
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")
tokenizer = Tokenizer(nlp.vocab)
nlp.add_pipe("emoji", first=True)

f = open('training/EXIST2023_training.json', 'r')
training = json.loads(f.read())

print('working')

#print(training)

for id in training:
	for value in training[id]:
		if value == 'tweet':
			tweet = (training[id][value])
			doc = nlp(tweet)
			for token in doc:
				if token._.is_emoji:
					#print(tweet[letter])
					training[id][value] = tweet.replace(token.text, token._.emoji_desc)
				if "@" in token.text:
					training[id][value] = tweet.replace(token.text, '')
					print(training[id][value])

f.close()
