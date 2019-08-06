from __future__ import division
import os
import argparse
import logging
import time
import torch
from nltk.translate.bleu_score import sentence_bleu
from torch.optim.lr_scheduler import StepLR
import torchtext
import pandas as pd
import sys
sys.path.append(".")
print(os.getcwd())
ROOT=os.getcwd()
import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint
print(os.getcwd())
try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

TRAIN_PATH = ROOT+"/data/triple-data/train/data2.csv"

DEV_PATH = ROOT+"/data/triple-data/dev/data2.csv"
parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data', default='TRAIN_PATH')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data',default='DEV_PATH')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)
totaltime=0
print(os.path.exists(TRAIN_PATH))
if opt.load_checkpoint is not None:
    start_time =time.time()
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
    loading_time = start_time - time.time()
else:
    # Prepare dataset
    src = SourceField()
    tgt = TargetField()
    max_len = 50
    def len_filter(example):
        print(example.src)
        return len(example.src) <= max_len and len(example.tgt) <= max_len

    train = torchtext.data.TabularDataset(
        path=TRAIN_PATH, format='csv',
        fields=[('src', src), ('tgt', tgt)]
    )
    dev = torchtext.data.TabularDataset(
        path=DEV_PATH, format='csv',
        fields=[('src', src), ('tgt', tgt)]
    )
    src.build_vocab(train, max_size=50000)
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size=128
        bidirectional = True
        encoder = EncoderRNN(len(src.vocab), max_len, hidden_size,
                             bidirectional=bidirectional, variable_lengths=True)
        decoder = DecoderRNN(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                             dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                             eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        seq2seq = Seq2seq(encoder, decoder)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    t = SupervisedTrainer(loss=loss, batch_size=32,
                          checkpoint_every=50,
                          print_every=10, expt_dir=opt.expt_dir)

    seq2seq = t.train(seq2seq, train,
                      num_epochs=100, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)
pred_start_time  = time.time()
predictor = Predictor(seq2seq, input_vocab, output_vocab)
pred_time = pred_start_time - time.time()
print("Time taken for inference is",loading_time+pred_time)

data = pd.read_csv( ROOT+"/data/triple-data/test/test.csv",delimiter=",",header=0)
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
    #target_str = data.iloc[i]["tgt"]
    print(seq_str)
    seq = seq_str.strip().split()
    pred.append(predictor.predict(seq))


print(pred)
score =0
pred_target= sentence_gen(pred)
print(len(pred_target))
for i in range(len(data)):
    score += sentence_bleu(pred_target[i], data.iloc[i]["tgt"], weights=(1, 0, 0, 0))
score = score/len(data)
pred_target = pd.DataFrame(pred_target)

pred_target.columns = ["pred"]
pred_target.to_csv("output.csv",sep=",")