from nltk.corpus import wordnet
import nltk
import numpy
import re
import runs
import hyperParameters
import graph
from utils import split_shakespeare, load_model
import metrics


class Augment_data():

    def __init__(self):
        DIR = "../data/shakespeare"
        file = open(f"{DIR}/shakespeare.txt", mode='r', encoding='utf-8').read()
        self.words = file.split(" ")
        self.create_txt_file(words=file)

    def create_txt_file(self, words):
        f = open("../data/shakespeare/augmented_shakespeare.txt", "w+")
        f.write(words) #self.words by default
        f.write("/n")
        f.close()

    def create_train_test_files(self):
        split_shakespeare(file="/augmented_shakespeare.txt")

    def synonym_replacement(self, percentage=20):
        words_replaced = 0
        total_words = len(self.words)
        must_change = False
        for idx, word in enumerate(self.words):
            chance = numpy.random.randint(0,101)
            if chance <= percentage or must_change:
                word = re.sub(r'[^a-zA-Z]', '', word)
                list_of_synonyms = self.get_synonym(word=word)
                if len(list_of_synonyms) != 0 and len(word) > 2:
                    selected_synonym = numpy.random.choice(list_of_synonyms)
                    #print(str(selected_synonym) + " FOR " + str(word))
                    #print("")
                    self.words[idx] = selected_synonym
                    words_replaced += 1
                    must_change = False
                else:
                    must_change = True
        self.words = " ".join(self.words)
        print("")
        print("words_replaced", words_replaced)
        print("total words", total_words)
        self.create_txt_file(words=self.words)



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

    def __init__(self, num_epchs=12000, m=200):
        self.num_epchs = num_epchs
        self.m = m
        self.plot = graph.Plot()
        self.run_augmented()
        self.run_normal()
        self.plot.show_graphs()
        print("")
        print("perp_aug", self.perp_aug)
        print("Perp_nor", self.perp_nor)

    def run_augmented(self):
        print("Starting run on Augmented Data")
        run = runs.Run(training_file="/train_aug_shakespeare.txt")
        h = hyperParameters.Hyper_params(hidden_size=self.m, num_epochs=self.num_epchs, dir = "../results/data_augmentation")
        self.lstm_aug = run.run_lstm(hyper_params=h, print_every=2000, return_model=True, fileName="aug")
        aug_seq = self.lstm_aug.generate(initial_str="t", generated_seq_length=400, temperature=0.5)
        self.perp_aug = metrics.getPerplexity(modelFile="../data/bigrams/testBigramsShakespeare.txt", generatedSequence=aug_seq)
        self.plot_graph(lstm=self.lstm_aug, y_leg="Trained on Augmented Data")

    def run_normal(self):
        print("Starting run on Augmented Data")
        run= runs.Run()
        h = hyperParameters.Hyper_params(hidden_size=self.m, num_epochs=self.num_epchs, dir = "../results/data_augmentation")
        self.lstm_normal = run.run_lstm(hyper_params=h, print_every=2000, return_model=True, fileName="nor")
        aug_seq = self.lstm_normal.generate(initial_str="t", generated_seq_length=400, temperature=0.5)
        self.perp_nor = metrics.getPerplexity(modelFile="../data/bigrams/testBigramsShakespeare.txt", generatedSequence=aug_seq)
        print("Perp_nor", self.perp_nor)
        self.plot_graph(lstm=self.lstm_normal, y_leg="Trained on Non-Augmented Data", color="Red")

    def plot_graph(self, lstm, y_leg, color="Blue"):
        print("DEBUG", lstm.history["loss"])
        self.plot.add_graphs(y=lstm.history["loss"], y_legend=y_leg, y_label="Loss", x_label="Per 100 Iterations", color=color)

class Validation:

    def __init__(self):
        self.perp_aug = []
        self.perp_nor = []
        for i in range(30):
            self.perp_augmented()
            self.perp_norm()
        self.perp_aug = numpy.mean(self.perp_aug)
        self.perp_nor = numpy.mean(self.perp_nor)
        print("perp_aug", self.perp_aug)
        print("Perp_nor", self.perp_nor)

    def perp_augmented(self):
        augmented_model = load_model(dir="../results/data_augmentation/lstm_hidden200_epoch12000_lr0.01_nlayer2aug.pth", hidden_size=200, num_layers=2, file="/train_aug_shakespeare.txt")
        aug_seq = augmented_model.generate(initial_str="t", generated_seq_length=400, temperature=0.5)
        #aug_seq = "And what so he for the beater heaver that we contrief, And the come the such come of the courther the counter the wear his counter the conster so the shall a so come the will what down the counted the present you me the be so the good the shall so whence the coult to the take the grain the consider for the state be the counter the counted the down the bed so so shall what to the stare"
        self.perp_aug.append(metrics.getPerplexity(modelFile="../data/bigrams/test_aug_shakespeare.txt", generatedSequence=aug_seq))

    def perp_norm(self):
        augmented_model = load_model(dir="../results/data_augmentation/lstm_hidden200_epoch12000_lr0.01_nlayer2nor.pth", hidden_size=200, num_layers=2)
        aug_seq = augmented_model.generate(initial_str="t", generated_seq_length=400, temperature=0.5)
        self.perp_nor.append(metrics.getPerplexity(modelFile="../data/bigrams/testBigramsShakespeare.txt", generatedSequence=aug_seq))

#norm
#F the should shall them with the seent the son to should the soul and the soul the will the some to shall and the soul the lies
#The love than the some the shall thou have will to the should speak and thou are the some the proce shall the stranger of stranger to the king the true and what thou come to should some words to stronger the son to speak the come to the strengthome of the some.

#aug_seq
#And what so he for the beater heaver that we contrief,
#And the come the such come of the courther the counter the wear his counter the conster so the shall a so come the will what down the counted the present you me the be so the good the shall so whence the coult to the take the grain the consider for the state be the counter the counted the down the bed so so shall what to the stare


if __name__ == '__main__':
    # Create Augment_data
    #d = Augment_data()
    #d.synonym_replacement()
    #d.create_train_test_files()
    test = Data_augmentation_tests()
    #Validation()
