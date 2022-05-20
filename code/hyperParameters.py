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
