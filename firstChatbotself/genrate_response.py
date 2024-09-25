import nltk
"""
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
"""
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import pickle
import numpy as np

# Load model
from tensorflow.keras.models import load_model
model = load_model('chatbot_model.h5')

import json
import random

intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words_lemma.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

'''def decrypt(msg):
    # i/p: what + is + machine + learning
    # o/p: what is machine learning
    # Remove '+' and replace it by ' '
    string = msg
    new_string = string.replace('+', ' ')
    return new_string'''

# Lemma input
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Create BOW of input
def bow(decrypt_msg, words, show_details=True):
    # Tokenize the pattern
    sentence_words = clean_up_sentence(decrypt_msg)
    
    # Bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            # Assign 1 if current word is in the vocabulary position
            if w == s:
                bag[i] = 1
                
                if show_details:
                    print ("Found in bag: %s" % w)
    
    return (np.array(bag))

# Use model to predict
def predict_class(decrypt_msg, model):
    # Filter out predictions below a threshold
    p = bow(decrypt_msg, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[idx, prob] for idx, prob in enumerate(res) if prob > ERROR_THRESHOLD]
    
    # Sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for rate in results:
        return_list.append({"intent": classes[rate[0]], "probability": str(rate[1])})
    
    return return_list

# Get response
def getResponse(pred_ints, intents_json):
    tag = pred_ints[0]['intent']
    list_of_intents = intents_json['intents']
    
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    
    return result

def chatbot_response(decrypt_msg):
    pred_ints = predict_class(decrypt_msg, model)
    res = getResponse(pred_ints, intents)
    return res

print(chatbot_response('what is machine learning'))