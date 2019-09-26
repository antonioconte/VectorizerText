from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import spacy
from preprocess.text_pipeline import TextPipeline

nlp = spacy.load('en_core_web_sm')
normalizer = TextPipeline(nlp)

from utils.metrics import metric


k = '1'
t = 'phrase'
path_data = '/home/anto/Scrivania/Tesi/testing/processed_data/' + t + '_' + k
with open(path_data, 'rb') as handle:
    data = pickle.load(handle)

X = [" ".join(item['data']) for item in data]
Y = [item['tag'] for item in data]
for i in range(550,2000):
    print(">",Y[i])
    print(">",X[i])
    print("\n ==== \n")
exit()

vectorizer = TfidfVectorizer(ngram_range=(3,3))
matrix = vectorizer.fit_transform(X)
knn = NearestNeighbors(n_neighbors=5,metric='cosine').fit(matrix)

def predict(id):
    query = Y[id]
    query_norm = normalizer.convert(query,divNGram=False)
    row = vectorizer.transform([query_norm])
    dist, idx = knn.kneighbors(row)
    idx = idx[0]
    for id in idx:
        res = metric(query_norm,Y[id],normalizer,Trigram=False)
        if float(res['lev']) >= 0.3:
            return float(res['lev'])

# test_idx = [10,20,30,40,50,60]
test_idx = [1]

print(vectorizer)
val = []
for id in test_idx:
    val += [predict(id)]
print('AVG: ',sum(val)/len(val))

vectorizer = None
knn = None

vectorizer = CountVectorizer(ngram_range=(3,3))
matrix = vectorizer.fit_transform(X)
knn = NearestNeighbors(n_neighbors=5,metric='cosine').fit(matrix)
print(vectorizer)

print()
val = []
for id in test_idx:
    val += [predict(id)]
print('AVG: ',sum(val)/len(val))

