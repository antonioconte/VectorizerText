from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import spacy
import time
from preprocess.text_pipeline import TextPipeline
from utils.metrics import metric
import config
class TextVectorizer:
    def __init__(self, model_type,part_type,ngram = 3):
        ngram_range = (ngram,ngram)
        self.part_type = part_type
        self.model_type = model_type
        if model_type == 'count':
            self.vectorizer = CountVectorizer(ngram_range=ngram_range)
        elif model_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range)
        else:
            print('model is not valid')
            exit(1)

        self.knn = None
        self.nlp = spacy.load('en_core_web_sm')
        self.normalizer = TextPipeline(self.nlp)

        path_data = config.path_data + self.part_type + '_3'
        with open(path_data, 'rb') as handle:
            data = pickle.load(handle)

        # vectorizer prende in input una stringa e non una lista di token
        #self.data_train = [ item['normalized'] for item in data]
        self.labels = [item['tag'] for item in data]

        self.path_knn = config.path_knn.format(model_type,part_type,ngram)
        self.path_vec = config.path_model.format(model_type,part_type,ngram)
        self.entryname = "{}_{}_{}".format(str.upper(model_type),part_type,ngram)
        print(self.path_knn)
        print(self.path_vec)

    def save(self):
        with open(self.path_knn, 'wb') as handle:
            pickle.dump(self.knn, handle)

        with open(self.path_vec, 'wb') as handle:
            pickle.dump(self.vectorizer, handle)



    def load(self):
        with open(self.path_knn, 'rb') as handle:
            self.knn = pickle.load(handle)

        with open(self.path_vec, 'rb') as handle:
            self.vectorizer = pickle.load(handle)

        self.l_sign = len(self.vectorizer.get_feature_names())
        print(self.vectorizer)

    def train(self):
        path_data = config.path_data + self.part_type + '_3'
        with open(path_data, 'rb') as handle:
            data = pickle.load(handle)

        # vectorizer prende in input una stringa e non una lista di token
        self.data_train = [item['normalized'] for item in data]

        print("==== TRAINING [ model = {}, part_type = {} ]".format(self.model_type,self.part_type))
        matrix = self.vectorizer.fit_transform(self.data_train)
        self.l_sign = len(self.vectorizer.get_feature_names())
        self.knn = NearestNeighbors(n_neighbors=config.num_recommendations, metric='cosine').fit(matrix)
        self.save()
        print("==== END ====================")

        pass

    def predict(self,
                query,
                threshold=config.default_threshold,
                N=config.num_recommendations,
                Trigram = False):

        start_time = time.time()

        query_norm = self.normalizer.convert(query, divNGram=False)
        query_vec = self.vectorizer.transform([query_norm])

        dist, idx = self.knn.kneighbors(query_vec)
        idx = idx[0]

        timing_search = "%.2f ms" % ((time.time() - start_time) * 1000)

        if len(idx) == 0:
            res_json = []
        else:
            res_json = []
            for id in idx:
                item = metric(query_norm, self.labels[id], self.normalizer, Trigram=Trigram)
                if float(item['lev']) >= threshold:
                    res_json += [item]
            # ====== RE-RANKING =========================================================
            res_json = sorted(res_json, key=lambda i: i['lev'], reverse=True)

        timing = "%.2f ms" % ((time.time() - start_time) * 1000)

        return {'query': query, 'data': res_json, 'time': timing, 'max': N, 'time_search': timing_search,
                'threshold': threshold, 'algoritm': self.entryname}

def train_only():
    model = 'tfidf'
    t = ['phrase','section','paragraph']
    for i in t:
        vect = TextVectorizer(model, i, ngram=3)
        vect.train()
    exit()

if __name__ == '__main__':

    # train_only()
    # exit()
    # vect = TextVectorizer('tfidf','paragraph',ngram=3)
    vect = TextVectorizer('tfidf','section',ngram=3)
    # vect = TextVectorizer('tfidf','phrase',ngram=3)

    vect.load()
    import json
    query = vect.labels[2014].split("]",1)[1]
    res = vect.predict(query)
    print(json.dumps(res,indent=4))
