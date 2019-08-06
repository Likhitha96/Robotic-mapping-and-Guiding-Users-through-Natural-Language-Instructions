import os
import sys
import numpy as np
import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd

def sen2triples(text):
	text=text.lower()
	wordsList = nltk.word_tokenize(text)
	tagged = nltk.pos_tag(wordsList)
	tagged1=[]
	#allowed = ['NN','IN','RB','NNS','VB','VBD','VBP','JJ']
	allowed = ['NN','IN','RB','NNS','NNPS','NNS','JJ']
	for tup in tagged:
		if tup[1] in allowed:
			tagged1.append(tup)
	#print(tagged)
	t=np.array(tagged1)
	#print(len(t))
	str=" ".join(t[:,0])
	return str
def readScan():
    df = []
    with open('../data/trainval.csv') as f:
        objData = pd.read_csv(f)
        objData = objData
        print(objData.shape)
        for i in range(len(objData)):
            #print("i is",i)
            string_list = objData.iloc[i]["tgt"].split()
            string_len = len(string_list)
            if string_len<30:
                src = sen2triples(objData.iloc[i]["tgt"])
                tgt = objData.iloc[i]["tgt"]
                path = objData.iloc[i]["path"]
                df.append([src,tgt,path])
        df = pd.DataFrame(df,columns = ["src","tgt","path"])
    return df
df = readScan()
print(df.shape)
df.to_csv("../data/traindata.csv",sep=",")