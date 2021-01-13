#!/usr/bin/env python
# coding: utf-8

# In[103]:


from flask import Flask,request,jsonify,render_template
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re,string,nltk
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import re
from keras.models import model_from_json
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stem = PorterStemmer()
import time,datetime


# In[2]:

## replace with your mdel name

json_file = open('SentimentAnalysis.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("SentimentAnalysis.h5")
print("Loaded model from disk")


# In[114]:
## KEEP HTML FILES UNDER FOLDER NAMED TEMPLATES

app = Flask(__name__)
# @app.route('/prediction/<text_data>')

@app.route('/')
## welcome page
def sentiment():
    return render_template('index.html')

@app.route('/prediction',methods=['GET'])
def prediction():
	### YOUR CODE 
    if request.method == 'GET':

        text_data = request.args.get('text_data')
        
    print(text_data)
    labels= {'0':'Negative','1':'Positive'}
#     text = request.form('text_data')
    def remove_emoji(string):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251""]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', string)
    def clean_data(txt):
        txt = ' '.join([word for word in txt.split() if word not in re.findall("^@.+",word)])
        txt = ' '.join([word for word in txt.split() if word not in re.findall("^http.+|^www\..+|[a-zA-Z0-9]+[.][a-zA-Z]{2,4}$",word)])
        txt = ' '.join([word for word in txt.split() if word not in re.findall("^:.+|^;.+|^XD|^xD",word)])
        txt = remove_emoji(txt)
        txt = ''.join([word.lower() for word in txt if word not in string.punctuation])
        txt = ' '.join([stem.stem(word) for word in txt.split() if word not in stopwords.words('english')])
        txt = word_tokenize(txt)
        return txt
    text = clean_data(text_data)
    text_len = loaded_model.input_shape[1]
    token_obj = Tokenizer()
    token_obj.fit_on_texts(text)
    sequences = token_obj.texts_to_sequences([text])
    word_index = token_obj.word_index
    seq_pad = pad_sequences(sequences,maxlen=text_len)
    predict = loaded_model.predict_classes(seq_pad)
    #out = {'Sentiment of text is ':labels[str(predict[0][0])]}
    #     print(jsonify(labels[str(predict[0][0])]))
#     result = labels[str(predict[0][0])]
    return render_template("output.html",result = labels[str(predict[0][0])]) 

  
      


# In[115]:


app.run(debug=True,use_reloader=False)


# In[ ]:





# In[ ]:




