from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory  # Rumus Library
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
import pandas as pd
import re

stemmer = StemmerFactory().create_stemmer()  # Object stemmer
remover = StopWordRemoverFactory().create_stop_word_remover()  # objek stopword

class DiceDistance(object):
    """
        Create custom libraries
    """
    def __init__(self, dataset):
    	self.dataset = dataset
    	self.RELEVANT = "Relevant"
    	self.IRRELEVANT = "Irrelevant"
    	self.LIST_REPLACE_TEXT = ['/', 'gram', 'ml', 'cc', 'buah', 'sendok teh', 'sendok makan', 'sendok takar', 'butir', 'cangkir', 'siung', 'batang']
    	self.scores = list()

    def replaceMultiple(self, mainString, toBeReplaces, newString):
	    # Iterate over the strings to be replaced
	    for elem in toBeReplaces:
	        # Check if string is in the main string
	        if elem in mainString:
	            # Replace the string
	            mainString = mainString.replace(elem, newString)
	    return mainString

    def preprocessing(self, documents):
    	cleanText = []
    	for text in documents:
    		lowerText = text.lower()
    		s = re.sub(r'[^\w\s]', '', lowerText)  # normalisasi text dari code
    		textStemmed = stemmer.stem(s)  # steming kata
    		textClean = remover.remove(textStemmed)  # membuang kata tdak penting
    		result = ''.join([i for i in textClean if not i.isdigit()])
    		otherStr = self.replaceMultiple(result, self.LIST_REPLACE_TEXT, "")
    		a = re.sub(' +', ' ', otherStr)  # untuk menghilangkan double spasi dikalimat
    		b = a.lstrip()  # untuk hilangkan spasi di depan kalimat
    		cleanText.append(b)
    	return cleanText

    def dice_distance(self, a, b):
    	c = a.intersection(b)
    	return float(2 * len(c)) / (len(a) + len(b))

    def search(self, query):
    	alldata = []
    	ingredients = self.preprocessing(self.dataset.Bahan)
    	preprocessedQuery = self.preprocessing([query])
    	alldata.append(set(preprocessedQuery[0].split(" ")))
    	for item in ingredients:
    		preprocessed = self.preprocessing([item])
    		alldata.append(preprocessed[0].split(" "))
    	for i in range(1, len(alldata)):
    		self.scores.append(self.dice_distance(alldata[0], alldata[i]))
    	return self.scores

    def relevantChecker(self):
    	information = []
    	for score in self.scores:
    		if score >= 0.5:
    			information.append(self.RELEVANT)
    		else:
    			information.append(self.IRRELEVANT)
    	return information