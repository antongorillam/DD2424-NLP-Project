from spellchecker import SpellChecker
import re
spell = SpellChecker()

# find those words that may be misspelled
misspelled = spell.unknown(['let', 'us', 'wlak','on','the','groun'])

# for word in misspelled:
#     # Get the one `most likely` answer
#     print(spell.correction(word))

#     # Get a list of `likely` options
#     print(spell.candidates(word))
# print('hi' in spell)

def getSpellPercentage(genString):
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
