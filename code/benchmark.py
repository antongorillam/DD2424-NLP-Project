from utils import read_data
import torch
import pandas as pd
import lstm
import rnn

if __name__ == '__main__':
    data_dict = read_data("../data/The_Sun_Also_Rises.txt")
    text = data_dict["text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    DIR = "../results"
    SEQUENCE_LENGTH = 25
    BATCH_SIZE = 1
    NUM_EPOCHS = 35000
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    TEMPERATURE = 0.28
    LEARNING_RATE = 0.1
    LABEL_SMOOTHING = 0

    # rnn_gen =  rnn.Generator(
    #     input_string=text, 
    #     index2char=index2char, 
    #     char2index=char2index,
    #     sequence_length=SEQUENCE_LENGTH,
    #     batch_size=BATCH_SIZE
    #     )

    # rnn =  rnn.RNN(
    #     input_size=len(index2char), 
    #     hidden_size=HIDDEN_SIZE, 
    #     num_layers=NUM_LAYERS, 
    #     output_size=len(index2char),
    # ).to(rnn_gen.device)

    # rnn_gen.train(
    #     rnn=rnn,
    #     num_epchs=NUM_EPOCHS,
    #     print_every=100,
    #     lr=LEARNING_RATE,
    #     temperature=TEMPERATURE,
    #     label_smoothing=LABEL_SMOOTHING,
    # )
    
    # rnn_df = pd.DataFrame(rnn_gen.history)
    # rnn_df.to_csv(f'{DIR}/rnn.csv', index=False) # TODO: add model configs to filename

    lstm_gen = lstm.Generator(
        input_string=text,
        index2char=index2char, 
        char2index=char2index,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
        )

    lstm_model = lstm.RNN(
        input_size=len(index2char), 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        output_size=len(index2char),
    ).to(lstm_gen.device)

    lstm_gen.train(
        lstm=lstm_model,
        num_epchs=NUM_EPOCHS,
        print_every=100,
        lr=LEARNING_RATE,
        temperature=TEMPERATURE,
        label_smoothing=LABEL_SMOOTHING,
    )

    """
    Save LSTM model
    """
    torch.save(lstm_gen.lstm.state_dict(), f"{DIR}/lstm_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.pth")
    lstm_df = pd.DataFrame(lstm_gen.history)
    lstm_df.to_csv(f'{DIR}/lstm_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.csv', index=False) 

    """
    To load model to cuda but it was saved on cpu (or vice versa) use:
    device = torch.device("cuda")
    lstm = lstm.RNN(*args, **kwargs)
    lstm.load_state_dict(torch.loadc(PATH, map_location=device))
    """
