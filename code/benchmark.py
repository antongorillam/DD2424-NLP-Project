from utils import read_data
import torch
import pandas as pd
import numpy as np
import lstm
import rnn
import time
from jury import Jury


"""
Deciding on what metrics to use
"""
if __name__ == '__main__':

    DIR_TRAIN = "../data/The_Sun_Also_Rises.txt"
    DIR_TEST = "../data/Old_Man_And_The_Sea.txt"
    
    data_dict = read_data(DIR_TRAIN, DIR_TEST)
    text = data_dict["train_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    DIR = "../results"
    SEQUENCE_LENGTH = 25
    BATCH_SIZE = 1
    NUM_EPOCHS = 10000
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    TEMPERATURE = 0.28
    LEARNING_RATE = 0.01
    LABEL_SMOOTHING = 0.8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    
    lstm_model =  lstm.RNN(
        input_size=len(index2char), 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        output_size=len(index2char),
    )

    lstm_model.load_state_dict(torch.load(f"{DIR}/lstm_epoch10000_lr0.01_nlayer2.pth", map_location=device))

    lstm_gen = lstm.Generator(
        input_string=text,
        index2char=index2char, 
        char2index=char2index,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )

    lstm_gen.lstm = lstm_model 
    TEMPERATURES = np.arange(0.1, 1, 0.1)   
    lstm_gen.generate(generated_seq_length=100)
    for temp in TEMPERATURES:
        print(f"temp {temp:.1f}: {lstm_gen.generate(generated_seq_length=100, temperature=temp)}\n")

    
    # toc = time.perf_counter()
    # scorer = Jury(metrics=['bleu','rouge','bertscore'])
    # predictions = [gen]
    # references = [text]
    # score = scorer.evaluate(predictions=predictions, references=references)
    # time_elapsed_sec = time.perf_counter() - toc
    # time_elapsed = time.strftime("%Hh:%Mm:%Ss", time.gmtime(time_elapsed_sec))
    # print(f'score: {score}')
    # print(f'Time elapsed: {time_elapsed}') 
