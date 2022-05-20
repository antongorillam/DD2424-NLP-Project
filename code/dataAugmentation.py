from nltk.corpus import wordnet
import nltk
import numpy
import re
import runs
import hyperParameters
import graph


class Augment_data():

    def __init__(self):
        DIR = "../data/shakespeare"
        file = open(f"{DIR}/train_shakespeare.txt", mode='r', encoding='utf-8').read()
        self.words = file.split(" ")

    def create_txt_file(self):
        f = open("../data/shakespeare/augmented_shakespeare_train.txt", "w+")
        f.write(self.words)
        f.close()

    def synonym_replacement(self, percentage=15):
        counter = 0
        percentage = int(100/percentage)
        words_replaced = 0
        total_words = len(self.words)
        for idx, word in enumerate(self.words):
            if counter >= percentage:
                word = re.sub(r'[^a-zA-Z]', '', word)
                list_of_synonyms = self.get_synonym(word=word)
                if len(list_of_synonyms) != 0 and len(word) > 2:
                    counter = 0
                    selected_synonym = numpy.random.choice(list_of_synonyms)
                    #print(str(selected_synonym) + " FOR " + str(word))
                    #print("")
                    self.words[idx] = selected_synonym
                    words_replaced += 1
            counter += 1

        self.words = " ".join(self.words)
        print("")
        print("words_replaced", words_replaced)
        print("total words", total_words)



    def get_synonym(self, word):
        nltk.data.path.append('./nltk_data/')
        synonyms = set()
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                synonym = l.name().replace("_", " ").replace("-", " ").lower()
                synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        synonyms = list(synonyms)
        return synonyms

class Data_augmentation_tests:

    def __init__(self, num_epchs=5000, m=200):
        self.num_epchs = num_epchs
        self.m = m
        self.plot = graph.Plot()
        self.run_augmented()
        self.run_normal()
        self.plot.show_graphs()

    def run_augmented(self):
        print("Starting run on Augmented Data")
        run = runs.Run(training_file="/augmented_shakespeare_train.txt")
        h = hyperParameters.Hyper_params(hidden_size=self.m, num_epochs=self.num_epchs, dir = "../results/data_augmentation")
        self.lstm_aug = run.run_lstm(hyper_params=h, print_every=500, return_model=True)
        self.plot_graph(lstm=self.lstm_aug, y_leg="Trained on Augmented Data")

    def run_normal(self):
        print("Starting run on Augmented Data")
        run= runs.Run()
        h = hyperParameters.Hyper_params(hidden_size=self.m, num_epochs=self.num_epchs, dir = "../results/data_augmentation")
        self.lstm_normal = run.run_lstm(hyper_params=h, print_every=500, return_model=True)
        self.plot_graph(lstm=self.lstm_normal, y_leg="Trained on Non-Augmented Data", color="Red")

    def plot_graph(self, lstm, y_leg, color="Blue"):
        print("DEBUG", lstm.history["loss"])
        self.plot.add_graphs(y=lstm.history["loss"], y_legend=y_leg, y_label="Loss", x_label="Per 100 Iterations", color=color)

if __name__ == '__main__':
    # Create Augment_data
    #Augment_data().synonym_replacement().create_txt_file()

    test = Data_augmentation_tests()
