# VectorizerText
```python
    ''' 
      @model_type = {'count', 'tfidf'}
      @part_type = {'phrase','section','paragraph'}
    '''class TextVectorizer:

    def __init__(self, model_type,part_type,ngram = 3):
        ...

```


### INSTALLAZIONE
1. mkdir DIR 
2. git clone https://github.com/antonioconte/VectorizerText.git DIR
3. virtualenv -p python3.5 venv
4. source venv/bin/activate
5. pip --no-cache-dir install -r requirements.txt && pip install nltk
6. python3 -m spacy download en_core_web_sm
7. python app.py

virtualenv -p python3.5 venv && source venv/bin/activate && pip --no-cache-dir install -r requirements.txt  && python3 -m spacy download en_core_web_sm && pip install nltk && echo $(hostname -i)
&& python app.py