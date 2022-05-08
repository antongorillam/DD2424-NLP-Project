from utils import  read_data
import lstm
import rnn

if __name__ == '__main__':
    data_dict = read_data("../data/The_Sun_Also_Rises.txt")
    text = data_dict["text"]
    index2char = data_dict["index2char"]
    char2index = data_dict["char2index"]
    SEQUENCE_LENGTH = 25
    BATCH_SIZE = 1
    NUM_EPOCHS = 10000
    HIDDEN_SIZE = 100
    NUM_LAYERS = 2
    TEMPERATURE = 0.28
    LEARNING_RATE = 0.01

    rnn_gen =  rnn.Generator(
        input_string=text, 
        index2char=index2char, 
        char2index=char2index,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
        )

    rnn =  rnn.RNN(
        input_size=len(index2char), 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        output_size=len(index2char),
    ).to(rnn_gen.device)

    rnn_gen.train(
        rnn=lstm,
        num_epchs=NUM_EPOCHS,
        print_every=100,
        lr=LEARNING_RATE,
        temperature=TEMPERATURE,
    )
    
    lstm_gen = lstm.Generator(
        input_string=text, 
        index2char=index2char, 
        char2index=char2index,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
        )

    lstm = lstm.RNN(
        input_size=len(index2char), 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        output_size=len(index2char),
    ).to(lstm_gen.device)

    lstm_gen.train(
        lstm=lstm,
        num_epchs=NUM_EPOCHS,
        print_every=100,
        lr=LEARNING_RATE,
        temperature=TEMPERATURE,
    )