# find the vocab set to use for feature selection
def findSet(file):
    vocab_set = ['THE_SET_OF_WORDS_TO_USE_AS_VOCAB']

    if file == 'VOCAB_SET_TO_FIND':
        return vocab_set
    else:
        return "ERROR: CANNOT_FIND_VOCAB_SET"