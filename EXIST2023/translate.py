from googletrans import Translator
import spacy
import json

f = open("training/EXIST2023_training.json", "r")
training = json.loads(f.read())

translator = Translator()

for id in training:
	for value in training[id]:
		if(value == "tweet"):
			#print(type(training[id][value]))
			training[id][value] = translator.translate(training[id][value], src="es").text
print(training)

f.close()
