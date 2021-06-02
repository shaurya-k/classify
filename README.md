create_vocabulary(directory, cutoff) -- create and return a vocabulary as a list of word types with counts >= cutoff in the training directory

create_bow(vocab, filepath) -- create and return a bag of words Python dictionary from a single document

load_training_data(vocab, directory) -- create and return training set (bag of words Python dictionary + label) from the files in a training directory

prior(training_data, label_list) -- given a training set, estimate and return the prior probability P(label) of each label

p_word_given_label(vocab, training_data, label) -- given a training set and a vocabulary, estimate and return the class conditional distribution 𝑃(𝑤𝑜𝑟𝑑∣𝑙𝑎𝑏𝑒𝑙) over all words for the given label using smoothing

train(training_directory, cutoff) -- load the training data, estimate the prior distribution P(label) and class conditional distributions 𝑃(𝑤𝑜𝑟𝑑∣𝑙𝑎𝑏𝑒𝑙), return the trained model

classify(model, filepath) -- given a trained model, predict the label for the test document (see below for implementation details including the return value)
