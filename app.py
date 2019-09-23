from flask_cors import CORS
from flask import Flask,request
import config
import json
from model import TextVectorizer
Vectorizer_f = Vectorizer_p = Vectorizer_s = Vectorizer_t = None

app = Flask(__name__)


CORS(app)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

@app.route('/', methods=['GET'])
def index():
	response = app.response_class(
		response=json.dumps({'data': "Welcome TF-IDF Entrypoint!"}, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response


@app.route('/query/', methods=['POST'])
def query():
	try:
		query = request.json['data']
		type = request.json['type']
	except:
		return app.response_class(
			response=json.dumps({'error':'query or type is empty'}, indent=4),
			status=505,
			mimetype='application/json'
		)

	try:
		threshold = request.json['threshold']
	except:
		threshold = config.default_threshold

	try:
		maxResults = request.json['max']
	except:
		maxResults = config.num_recommendations
	Vectorizer_m = None
	T = False
	if type == "Phrase":
		Vectorizer_m = Vectorizer_f
	elif type == "Paragraph":
		Vectorizer_m = Vectorizer_p
	elif type == "Section":
		Vectorizer_m = Vectorizer_s
	elif type == "TriGram":
		T = True
		Vectorizer_m = Vectorizer_t

	if Vectorizer_m == None:
		return app.response_class(
			response=json.dumps({'error': 'type is not valid'}, indent=4),
			status=505,
			mimetype='application/json'
		)

	result = Vectorizer_m.predict(query,threshold=threshold,N=maxResults,Trigram=T)

	response = app.response_class(
		response=json.dumps(result, indent=4),
		status=200,
		mimetype='application/json'
	)
	return response

@app.route('/connect/', methods=['GET'])
def connect():
	k = str(request.args.get('k', default=3, type=int))
	global Vectorizer_f
	global Vectorizer_p
	global Vectorizer_s
	global Vectorizer_t

	models = []
	msg = "NOT GOOD"
	# load model phrase

	Vectorizer_f = TextVectorizer('tfidf', 'phrase', ngram=3)
	Vectorizer_f.load()
	models.append("Phrase_"+k)

	Vectorizer_p = TextVectorizer('tfidf', 'paragraph', ngram=3)
	Vectorizer_p.load()
	models.append("Paragraph_"+k)

	Vectorizer_s = TextVectorizer('tfidf', 'section', ngram=3)
	Vectorizer_s.load()
	models.append("Section_"+k)

	# Minhash_t = Minhash('trigram',k=k)
	# Minhash_t.load()
	# models.append("TriGram")


	if len(models) ==  4:
		msg = "All loaded!"
	elif len(models) > 0:
		msg = "loaded"

	response = app.response_class(
		response=json.dumps({
			'data': msg,
			'K': k,
			'models': models,
			'path_vectorizer':config.path_model,
			'path_knn': config.path_knn,
			'path_data': config.path_data,
			'wordbased': config.wordBased,
			'ip': config.ip
		}, indent=4, sort_keys=True),
		status=200,
		mimetype='application/json'
	)
	return response

if __name__ == '__main__':
	app.run('0.0.0.0')
 #port="1234"