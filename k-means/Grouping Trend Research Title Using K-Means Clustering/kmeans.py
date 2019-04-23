import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

class Clusterizing:
    def __init__(self, documents, n_clusters):
        self.documents = documents
        self.n_clusters = n_clusters

    @staticmethod
    def fit(documents):
        vocabulary = set([])
        for doc in documents:
            vocabulary.update(doc)
        vocabulary = list(vocabulary)

        features = np.zeros((len(documents), len(vocabulary)))
        for index, doc in enumerate(documents):
            for word in doc:
                if word in vocabulary:
                    column = vocabulary.index(word)
                    features[index, column] += 1

        return features

    def getClusters(self):
        features = self.fit(self.documents)
        cosine = cosine_similarity(features)

        kmeans = KMeans(n_clusters=self.n_clusters, max_iter=300, random_state=0).fit(cosine)
        return kmeans.labels_

    def getNClusters(self):
        return self.n_clusters