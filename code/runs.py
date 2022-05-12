from utils import read_data
import torch
import pandas as pd
import lstm
import rnn
import seaborn as sns
import matplotlib.pyplot as plt

class Hyper_params:

    # Defines the hyper parameters of the RNN
    def __init__(self, seq_length=100, batch_size=1, num_epochs = 100000, hidden_size = 100, num_layers = 2, temp = 0.29, learning_rate = 0.01, label_smoothing = 0, dir = "../results/rnn_vs_lstm" ):
        self.SEQUENCE_LENGTH = seq_length
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = num_epochs
        self.HIDDEN_SIZE = hidden_size
        self.NUM_LAYERS = num_layers
        self.TEMPERATURE = temp
        self.LEARNING_RATE = learning_rate
        self.LABEL_SMOOTHING = label_smoothing
        self.DIR = dir


    # Set 1 to run the LSTM
def run_lstm(hyper_params):

    print("Creating a Generator")
    lstm_gen = lstm.Generator(
    input_string=train_text,
    test_string=test_text,
    index2char=index2char,
    char2index=char2index,
    sequence_length=hyper_params.SEQUENCE_LENGTH,
    batch_size=hyper_params.BATCH_SIZE
    )
    print("Init RNN")
    lstm_model = lstm.RNN(
        input_size=len(index2char),
        hidden_size=hyper_params.HIDDEN_SIZE,
        num_layers=hyper_params.NUM_LAYERS,
        output_size=len(index2char),
    ).to(lstm_gen.device)

    print("Train")
    lstm_gen.train(
        lstm=lstm_model,
        num_epchs=hyper_params.NUM_EPOCHS,
        print_every=500,
        lr=hyper_params.LEARNING_RATE,
        temperature=hyper_params.TEMPERATURE,
        label_smoothing=hyper_params.LABEL_SMOOTHING,
    )


    """
    Save LSTM model
    """
    torch.save(lstm_gen.lstm.state_dict(), f"{DIR}/lstm_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.pth")
    lstm_df = pd.DataFrame(lstm_gen.history)
    lstm_df.to_csv(f'{DIR}/lstm_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.csv', index=False)

    sns.set_style("whitegrid")
    title_string = f"LSTM: Loss vs iterations\nHidden Layers:{HIDDEN_SIZE}, lr:{LEARNING_RATE}"
    lstm_fig = sns.lineplot(data=lstm_df, x="iterations", y="loss")
    lstm_fig.get_figure().savefig(f'{DIR}/lstm_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.png')
    """
    To load model to cuda but it was saved on cpu (or vice versa) use:
    device = torch.device("cuda")
    lstm = lstm.RNN(*args, **kwargs)
    lstm.load_state_dict(torch.loadc(PATH, map_location=device))
    """

def run_rnn(hyper_params):
    rnn_gen =  rnn.Generator(
        input_string=train_text,
        test_string=test_text,
        index2char=index2char,
        char2index=char2index,
        sequence_length=hyper_params.SEQUENCE_LENGTH,
        batch_size=hyper_params.BATCH_SIZE
        )

    rnn_model =  rnn.RNN(
        input_size=len(index2char),
        hidden_size=hyper_params.HIDDEN_SIZE,
        num_layers=hyper_params.NUM_LAYERS,
        output_size=len(index2char),
    ).to(rnn_gen.device)

    rnn_gen.train(
        rnn=rnn_model,
        num_epchs=hyper_params.NUM_EPOCHS,
        print_every=500,
        lr=hyper_params.LEARNING_RATE,
        temperature=hyper_params.TEMPERATURE,
        label_smoothing=hyper_params.LABEL_SMOOTHING,
    )

    """
    Save RNN model
    """
    torch.save(rnn_model.rnn.state_dict(), f"{DIR}/rnn_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.pth")
    rnn_df = pd.DataFrame(rnn_gen.history)
    rnn_df.to_csv(f"{DIR}/rnn_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.csv", index=False)

    sns.set_style("whitegrid")
    title_string = f"RNN: Loss vs iterations\nHidden Layers:{HIDDEN_SIZE}, lr:{LEARNING_RATE}"
    rnn_fig = sns.lineplot(data=rnn_df, x="iterations", y="loss")
    rnn_fig.get_figure().savefig(f'{DIR}/rnn_hidden{HIDDEN_SIZE}_epoch{NUM_EPOCHS}_lr{LEARNING_RATE}_nlayer{NUM_LAYERS}.png')

if __name__ == '__main__':
    data_dict = read_data()
    train_text = data_dict["train_text"]
    test_text = data_dict["test_text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]

    hyper_params = Hyper_params()

    run_lstm(hyper_params=hyper_params)
    # Set 1 to run the RNN
