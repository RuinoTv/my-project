from keras.src.layers import average
import nltk
 import re

def normal(text):
     text = text.lower()
     punctuation = r"[^\w\s]"
     return re.sub(punctuation, " ", text)

def get_rank(text1, text2):
     text1 = normal(text1)
     text2 = normal(text2)
     distance = nltk.edit_distance(text1, text2)
     average_length = (len(text1) + len(text2)) / 2
     return distance / average_length

question = input("сурак кой: ")

if get_rank(question, "как тебя завут?") < 0.4:
     print("никак")

if get_rank(question, "какого цвета облока?") < 0.4:
     print("белого")

if get_rank(question, "привет") < 0.4:
     print("привет")