#####################
#  classify.py
#  shaurya kethireddy
#####################
import os
import math


def create_bow(vocab, filepath):
    """ Create a single dictionary for the data
        Note: label may be None
    """
    bow = {None: 0}
    for var in vocab:
        bow[var] = 0

    file = open(filepath, "r")
    for line in file:
        holder = line.rstrip()
        if holder in bow:
            bow[holder] += 1
        else:
            bow[None] += 1

    if bow[None] == 0:
        del bow[None]
    for obj in vocab:
        if bow[obj] == 0:
            del bow[obj]
    return bow


def create_vocabulary(directory, cutoff):
    """ Create a vocabulary from the training directory
        return a sorted vocabulary list
    """
    top_level = os.listdir(directory)
    vocab = {}
    for d in top_level:
        subdir = d if d[-1] == '/' else d+'/'
        files = os.listdir(directory+subdir)
        for f in files:
            with open(directory+subdir+f, 'r') as doc:
                for word in doc:
                    word = word.strip()
                    if not word in vocab and len(word) > 0:
                        vocab[word] = 1
                    elif len(word) > 0:
                        vocab[word] += 1
    return sorted([word for word in vocab if vocab[word] >= cutoff])


def load_training_data(vocab, directory):
    """ Create the list of dictionaries """
    top_level = os.listdir(directory)
    dataset = []
    for d in top_level:
        if d[-1] == '/':
            label = d[:-1]
            subdir = d
        else:
            label = d
            subdir = d+"/"
        files = os.listdir(directory+subdir)
        for f in files:
            bow = create_bow(vocab, directory+subdir+f)
            dataset.append({'label': label, 'bow': bow})
    return dataset


def prior(training_data, label_list):
    """ return the prior probability of the label in the training set
        => frequency of DOCUMENTS
    """

    smooth = 1  # smoothing factor
    logprob = {}
    total = len(training_data)
    for label in label_list:
        ctr = 0
        for data in training_data:
            if data['label'] == label:
                ctr += 1
        # log probability of certain label
        logprob[label] = math.log((ctr + smooth) / float(total + len(label_list)))

    return logprob


def p_word_given_label(vocab, training_data, label):
    """ return the class conditional probability of label over all words, with smoothing """
    smooth = 1  # smoothing factor
    word_prob = {}
    dict = {None: 0}
    for it in vocab:
        dict[it] = 0
    ctr = 0
    for data in training_data:
        for word in data['bow']:
            if data['label'] == label:
                ctr += data['bow'][word]
    for data in training_data:
        if data['label'] == label:
            for word in data['bow']:  # Go through each valid bow
                if word in vocab:  # Word is either in vocab or part of None
                    dict[word] += data['bow'][word]
                else:
                    dict[None] += data['bow'][word]

    for x in dict:
        word_prob[x] = math.log((dict[x] + smooth) / float(ctr + smooth * (len(vocab) + 1)))

    return word_prob


def train(training_directory, cutoff):
    """ return a dictionary formatted as follows:
            {
             'vocabulary': <the training set vocabulary>,
             'log prior': <the output of prior()>,
             'log p(w|y=2016)': <the output of p_word_given_label() for 2016>,
             'log p(w|y=2020)': <the output of p_word_given_label() for 2020>
            }
    """
    retval = {}
    label_list = os.listdir(training_directory)
    vocab = create_vocabulary(training_directory, cutoff)
    data = load_training_data(vocab, training_directory)
    retval['vocabulary'] = vocab
    retval['log prior'] = prior(data, label_list)
    retval['log p(w|y=2016)'] = p_word_given_label(vocab, data, '2016')
    retval['log p(w|y=2020)'] = p_word_given_label(vocab, data, '2020')
    # print(retval)
    return retval


def classify(model, filepath):
    """ return a dictionary formatted as follows:
            {
             'predicted y': <'2016' or '2020'>,
             'log p(y=2016|x)': <log probability of 2016 label for the document>,
             'log p(y=2020|x)': <log probability of 2020 label for the document>
            }
    """
    retval = {}
    sixteen = 0
    twenty = 0
    file = open(filepath, "r")
    for line in file:
        holder = line.rstrip()
        if holder in model['vocabulary']:
            sixteen += model['log p(w|y=2016)'][holder]
            twenty += model['log p(w|y=2020)'][holder]
        else:
            sixteen += model['log p(w|y=2016)'][None]
            twenty += model['log p(w|y=2020)'][None]
    prob20 = model['log prior']['2020'] + twenty
    prob16 = model['log prior']['2016'] + sixteen
    retval['log p(y=2020|x)'] = prob20
    retval['log p(y=2016|x)'] = prob16
    if prob20 > prob16:
        retval['predicted y'] = '2020'
    else:
        retval['predicted y'] = '2016'
    return retval


# train('./corpus/test/', 2)
# vocab = create_vocabulary('./corpus/training/', 2)
# training_data = load_training_data(vocab,'./corpus/training/')
# print(prior(training_data, ['2020', '2016']))

# model = train('./corpus/training/', 2)
# classify(model, './corpus/test/2016/0.txt')


