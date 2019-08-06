import os
import argparse
import pandas as pd
import sys
sys.path.append(".")
print(os.getcwd())
ROOT=os.getcwd()

from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

INPUT_FILE = "/data/triple-data/test/test_1.csv"

parser = argparse.ArgumentParser()
parser.add_argument('--input', action='store', dest='input',
                    help='Input file', default='INPUT_FILE')
opt = parser.parse_args()

checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
checkpoint = Checkpoint.load(checkpoint_path)
seq2seq = checkpoint.model
input_vocab = checkpoint.input_vocab
output_vocab = checkpoint.output_vocab

predictor = Predictor(seq2seq, input_vocab, output_vocab)

data = pd.read_csv( ROOT+INPUT_FILE,delimiter=",",header=0)
pred=[]
def sentence_gen(sen):
    #sen[max_len,batch_size]
    a=[]

    for i in range(len(sen)):
        phrase = ""
        for j in range(len(sen[i])):
            if sen[i][j]!="<eos>" :
                #print("printing word :",sen[i][j])
                if sen[i][j]=="." or sen[i][j]==",":
                    phrase = phrase + sen[i][j]
                else:
                    phrase = phrase +" "+ sen[i][j]
            else:
                a.append(phrase)
                break
    return a

for i in range(len(data)):
    seq_str = data.iloc[i]["src"]
    print(seq_str)
    seq = seq_str.strip().split()
    pred.append(predictor.predict(seq))


print(pred)
pred_target= sentence_gen(pred)
print(len(pred_target))
pred_target = pd.DataFrame(pred_target)

pred_target.columns = ["pred"]
pred_target.to_csv("output.csv",sep=",")