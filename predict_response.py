import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
import json
import pickle
import numpy as np
import random
import tensorflow as tf
from data_preprocessing import get_stem_words

ignore_words = ['?', '!',',','.', "'s", "'m", "ñ", "ç"]
sla = tf.keras.models.load_model("./chatbot_model.h5")
intents = json.loads(open("./intents.json").read())
words = pickle.load(open("./words.pkl", "rb"))
clehss = pickle.load(open("./classes.pkl", "rb"))

def preprocess_user_input(user_input):

    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    bag=[]
    bag_of_words = []
   
    # Codificação dos dados de entrada 
    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
  
    return np.array(bag)

def boti_cless_prediction(user_input):
    duqtivelaimcima = preprocess_user_input(user_input)
    boladecristal = sla.predict(duqtivelaimcima)
    ousla = np.argmax(boladecristal[0])
    return(ousla)

def bot_response(user_input):

   predicted_class_label =  boti_cless_prediction(user_input)
   predicted_class = clehss[predicted_class_label]

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
        bot_response = random.choice(intent['responses'])
        return bot_response

print("Meu nome é yoshikage kira, como eu posso te ajudar?")

while True:
    user_input = input("Digite sua mensagem aqui:")
    print("Entrada do Usuário: ", user_input)

    response = bot_response(user_input)
    print("Resposta do Robô: ", response)
