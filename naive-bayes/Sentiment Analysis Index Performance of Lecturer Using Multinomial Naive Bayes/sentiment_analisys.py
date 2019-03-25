import pandas as pd
import numpy as np
import pickle
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

vectorizer = TfidfVectorizer()
naivebayes = MultinomialNB()
stemmer = StemmerFactory().create_stemmer()
remover = StopWordRemoverFactory().create_stop_word_remover()

class SentimentAnalysis(object):
    
    def __init__(self, train):
        self.loadStandardWords = pd.read_csv('standard_words.csv')
        self.datasets = pd.read_csv('dataset_raw.csv')
        self.train = train
        self.employeeId = [str(x) for x in list(train["pegawai_id_pegawai"])]
        self.teachId = [str(x) for x in list(train["ampu_id_ampu"])]
        self.standartWords = [str(x) for x in list(self.loadStandardWords["kata_baku"])]
        self.loadVector = None
        self.loadNB = None
        X = vectorizer.fit_transform(self.datasets.preprocessing_result.values.astype('U'))
        naivebayes.fit(X, self.datasets.sentimen)
        vectorFile = open('vectorizer.b', 'wb')
        nbFile = open('naive_bayes.b', 'wb')
        pickle.dump(vectorizer, vectorFile)
        pickle.dump(naivebayes, nbFile)
        vectorFile.close()  # close it to make sure it's all been written
        nbFile.close()  # close it to make sure it's all been written
        
    def getUniqueList(self, items):
        unique = list()
        for item in items:
            unique.append(item)
        return unique
    
    def loadPickle(self):
        self.loadVector = pickle.load(open('vectorizer.b', 'rb'), encoding='latin1')
        self.loadNB = pickle.load(open('naive_bayes.b', 'rb'), encoding='latin1')
        
    def preprocesing(self, documents):
        cleanText = list()
        for doc in documents:
            lowerText = doc.lower()
            stemmedText = stemmer.stem(lowerText)
            filteredText = remover.remove(stemmedText)
            for i, text in enumerate(self.loadStandardWords['vocabulary']):
                if filteredText == text:
                    filteredText = self.standartWords[i]
            cleanText.append(filteredText)
        return cleanText
    
    def predict(self):
        self.loadPickle() # Load pickle for vectorizer and naive bayes
        """
            Define variables
        """
        constructData = list()
        finalData = list()
        sentimentData = list()
        newSentimentData = list()
        saveIndex = list()
        total = 0
        sumPositive = 0
        sumNegative = 0
        sumNeutral = 0
        sumAll = 0
        totalPercentNeutral = 0
        totalPercentNegative = 0
        totalPercentPositive = 0
        answer = self.preprocesing(self.train.answer)
        """
            Get Prediction from Term Frequency
        """
        termFrequency = self.loadVector.transform(answer)
        for i in termFrequency:
            sentimentData.append(self.loadNB.predict(i))
            
        """
            Create custom data
        """
        for i, item in enumerate(answer):
            index = self.employeeId[i] + ',' + self.teachId[i]
            saveIndex.append(index)
            newSentimentData.append([index, [sentimentData[i]]])
        newSentimentData = self.getUniqueList(newSentimentData)
        cleanIndex = list(set(saveIndex))
        """
            Get Calculation
        """
        for i in cleanIndex:
            for j in newSentimentData:
                if str(i) == j[0]:
                    if np.int_(j[1]) == 0:
                        sumNegative += 1
                    elif np.int_(j[1]) == 2:
                        sumPositive += 1
                    elif np.int_(j[1]) == 1:
                        sumNeutral += 1
                    sumAll += 1
            percentNegative = round((float(sumNegative) / float(sumAll)) * 100, 2)
            percentNeutral = round((float(sumNeutral) / float(sumAll)) * 100, 2)
            percentPositive = round((float(sumPositive) / float(sumAll)) * 100, 2)
            totalPercentNegative += percentNegative
            totalPercentNeutral += percentNeutral
            totalPercentPositive += percentPositive
            constructData.append([i, percentPositive, percentNeutral, percentNegative])
        averagePositive = round(float(totalPercentPositive) / len(cleanIndex), 2)
        averageNeutral = round(float(totalPercentNeutral) / len(cleanIndex), 2)
        averageNegative = round(float(totalPercentPositive) / len(cleanIndex), 2)
        """
            Get final data that will ready used to dataframe
        """
        for i in constructData:
            n = i[0].split(',')
            dictionary = {
                'Id Dosen': n[0],
                'Id Mata Kuliah': n[1],
                'Sentimen Positif (%)': i[1],
                'Sentimen Netral (%)': i[2],
                'Sentimen Negatif (%)': i[3]
            }
            finalData.append(dictionary)
        return finalData