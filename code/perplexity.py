#  -*- coding: utf-8 -*-
import argparse
import codecs
import json
import math
from collections import defaultdict

import nltk
import numpy as np
import requests

"""
This file is part of the computer assignments for the course DD1418/DD2418 Language engineering at KTH.
Created 2017 by Johan Boye and Patrik Jonell.
"""


class BigramTester(object):
    def __init__(self):
        """
        This class reads a language model file and a test file, and computes
        the entropy of the latter.
        """
        # The mapping from words to identifiers.
        self.index = {}

        # The mapping from identifiers to words.
        self.word = {}

        # An array holding the unigram counts.
        self.unigram_count = {}

        # The bigram log-probabilities.
        self.bigram_prob = defaultdict(dict)

        # Number of unique words (word forms) in the training corpus.
        self.unique_words = 0

        # The total number of words in the training corpus.
        self.total_words = 0

        # The average log-probability (= the estimation of the entropy) of the test corpus.
        # Important that it is named self.logProb for the --check flag to work
        self.logProb = 0

        # The identifier of the previous word processed in the test corpus. Is -1 if the last word was unknown.
        self.last_index = -1

        # The fraction of the probability mass given to unknown words.
        self.lambda3 = 0.000001

        # The fraction of the probability mass given to unigram probabilities.
        self.lambda2 = 0.01 - self.lambda3

        # The fraction of the probability mass given to bigram probabilities.
        self.lambda1 = 0.99

        # The number of words processed in the test corpus.
        self.test_words_processed = 0

        self.sequence_iterator = 0

    def read_model(self, filename):
        """
        Reads the contents of the language model file into the appropriate data structures.

        :param filename: The name of the language model file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """

        try:
            with codecs.open(filename, "r", "utf-8") as f:
                self.unique_words, self.total_words = map(
                    int, f.readline().strip().split(" ")
                )
                # YOUR CODE HERE
                for i in range(self.unique_words):
                    idx, word, count = f.readline().strip().split(" ")
                    idx = int(idx)
                    count = int(count)
                    self.index[word] = idx
                    self.word[idx] = word
                    self.unigram_count[word] = count

                while True:
                    line = f.readline().strip()
                    if line == "-1":
                        # print('exit')
                        break
                    # print(line)
                    w1, w2, logprob = line.split(" ")
                    w1 = int(w1)
                    w2 = int(w2)
                    logprob = float(logprob)
                    self.bigram_prob[(w1, w2)] = logprob

                # for i in range(len(bigramLines)-1):
                #     w1, w2, logprob = bigramLines[i].strip().split(' ')

                return True
        except IOError:
            print("Couldn't find bigram probabilities file {}".format(filename))
            return False

    def compute_entropy_cumulatively(self, word):

        Prob = 0

        if word in self.index:

            if (self.last_index, self.index[word]) in self.bigram_prob:
                Prob += (
                    self.lambda1
                    * (math.exp(self.bigram_prob[(self.last_index, self.index[word])]))
                    + self.lambda2 * (self.unigram_count[word] / self.total_words)
                    + self.lambda3
                )
            else:
                Prob += self.lambda2 * (self.unigram_count[word] / self.total_words if word in self.index else 0 ) + self.lambda3
        else:
            Prob += self.lambda3

        if self.test_words_processed == 0:
            self.logProb = 1 / Prob
        else:
            self.logProb *= 1 / Prob
        
        self.test_words_processed += 1
        self.last_index = self.index.get(word)

        if self.test_words_processed == len(self.tokens):
            # print('logprob before',self.logProb)
            self.logProb = np.power(self.logProb, (1 / len(self.tokens)))
        
    def process_test_file(self, test_filename):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:
            with codecs.open(test_filename, "r", "utf-8") as f:
                self.tokens = nltk.word_tokenize(
                    f.read().lower()
                )  # Important that it is named self.tokens for the --check flag to work
                for token in self.tokens:
                    self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print("Error reading testfile")
            return False

    def process_test_string(self, test_string):
        """
        <p>Reads and processes the test file one word at a time. </p>

        :param test_filename: The name of the test corpus file.
        :return: <code>true</code> if the entire file could be processed, false otherwise.
        """
        try:

            self.tokens = nltk.word_tokenize(
                test_string.lower()
            )  # Important that it is named self.tokens for the --check flag to work
            for token in self.tokens:
                self.compute_entropy_cumulatively(token)
            return True
        except IOError:
            print("Error reading testfile")
            return False


def getPerplexity(modelFile, generatedSequence, type="string"):
    bigram_tester = BigramTester()
    bigram_tester.read_model(modelFile)
    if type == "file":
        bigram_tester.process_test_file(generatedSequence)
    else:
        bigram_tester.process_test_string(generatedSequence)
    return bigram_tester.logProb
