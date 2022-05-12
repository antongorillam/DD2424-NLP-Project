from spellchecker import SpellChecker
import re
from perplexity import BigramTester
import numpy as np

def getSpellPercentage(genString):
    spell = SpellChecker()
    #genString = re.sub(r"[^a-zA-Z\s]", "", genString)
    genString = re.sub(r"(?<=[A-Za-z])[”\.\,]", "", genString)
    genString = re.sub(r"(?=[A-Za-z])”", "", genString)
    genString = re.sub(r"\s{1}&\s{1}|(?<!\s)\?\s{1}", " ", genString)
    #print('regexed string: ', genString)
    genString = genString.lower().split()
    noWords = len(genString)
    correctcount = 0
    
    for word in genString:
        #print('word : ', word)
        #print('word : ', word,'in' if word in spell else 'not')
        if word in spell:
            correctcount += 1
        # else:
        #     print('word : ', word)
    return correctcount / noWords

def getPerplexity(modelFile, generatedSequence, type="string"):
    bigram_tester = BigramTester()
    bigram_tester.read_model(modelFile)
    if type == "file":
        bigram_tester.process_test_file(generatedSequence)
    else:
        bigram_tester.process_test_string(generatedSequence)
    return bigram_tester.logProb

def getAdjustedBLEU(candidate, reference):
    from jury import Jury
    
    scorer = Jury(metrics=['bleu'])
    prediction = [candidate]
    reference = [reference]
    score = scorer.evaluate(predictions=prediction, references=reference)
    nGramPrecisions = score['bleu']['precisions']
    precision = 1
    adjBLEU = {} 
    for i, prec in enumerate(nGramPrecisions):
        precision *= prec
        adjBLEU[i+1] = np.power(precision,(1/(i+1)))
    return adjBLEU, nGramPrecisions
