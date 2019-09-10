"""
Classify data.
"""
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from itertools import combinations
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pickle
def read_data(path):
    data = pd.read_csv(path)
    label = np.asarray(data['polarity'])
    docs = np.asarray(data['text'])
    return docs,label
def get_tweets(filename):
    """
    Load stored tweets.
    List of strings, one per tweet.
    """
    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)
def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    Params:
      doc....a string.

    Returns:
      a numpy array containing the resulting tokens.
"""

    if keep_internal_punct:
        return np.asarray([re.sub('^\W+|\W+$', '', w.lower()) for w in doc.split()])
    else:
        return np.asarray([w.lower() for w in re.findall('[\w]+', doc)])
    pass
def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.
"""
    for token in tokens:
        feats['token='+ token] +=1
    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.
    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    """
    ###TODO
    for i in range(0, len(tokens)-k+1):
        for com in combinations(tokens[i:i+k], 2):
            feats['token_pair='+com[0]+'__'+com[1]] += 1
    pass

def token_quadrigram_features(tokens, feats, k=5):
    for i in range(0,len(tokens)-k+1):
        for com in combinations(tokens[i:i+k], 4):
            feats['token_trigram_features='+ com[0]+'__'+com[1]+'__'+com[2]+'__'+com[3]] += 1
    pass



# afinn = read_afinn()
# def lexicon_score(tokens, label):
#         score = 0
#         for t in tokens:
#             if t in afinn.keys():
#                 score += afinn[t]
#         if score >0:
#             label.append(1)
#         else:
#             label.append(0)


# def lexicon_score1(tokens, feats):
#     feats['score'] = 0
#     for t in tokens:
#         if t in afinn.keys():
#             feats['score'] += afinn[t]
def featurize(tokens, feature_fns):
    """
    Compute all features for a list of tokens from
    a single document.

    Params:
      tokens........array of token strings from a document.
      feature_fns...a li
    Returns:st of functions, one per feature
      list of (feature, value) tuples, SORTED alphabetically
      by the feature name.
"""
    feats = defaultdict(lambda : 0)
    for fns in feature_fns:
        fns(tokens, feats)
    return sorted(feats.items(), key=lambda x:x[0])
    pass
def vectorize(tokens_list, feature_fns, min_freq, vocab=None):

    feats = []
    if vocab==None:
        vocab = {}
        id = 0
        freq = defaultdict(lambda: 0)

        for tokens in tokens_list:
            feature = featurize(tokens, feature_fns)
            feats.append(feature)
            for t in feature:
                freq[t[0]] += 1
        for k in sorted(freq.items(), key=lambda x: x[0]):
            if k[1] >= min_freq:
                vocab[k[0]] = id
                id += 1

    else:
        for tokens in tokens_list:
            feature = featurize(tokens, feature_fns)
            feats.append(feature)

    X_row = []
    X_column = []
    X_values = []

    for i in range(len(feats)):
        for t in feats[i]:
            if t[0] in vocab.keys():
                X_row.append(i)
                X_column.append(vocab[t[0]])
                X_values.append(t[1])

    X = csr_matrix((X_values,(X_row, X_column)), shape=(len(tokens_list),len(vocab)), dtype='int64')

    return X, vocab
    pass
def accuracy_score(truth, predicted):
    """ Compute accuracy of predictions.
    DONE ALREADY
    Params:
      truth.......array of true labels (0 or 1)
      predicted...array of predicted labels (0 or 1)
    """
    return len(np.where(truth==predicted)[0]) / len(truth)
neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])
def read_sentiment():
    """
    Read sentiment.txt into pos and neg words
    """
    with open('sentiment.txt', 'r') as f:
        for line in f:
            line = line.strip().split()
            if line[1]<0:
                neg_words.add(line[1])
            elif line[1]>0:
                pos_words.add(line[1])
def lexicon_features(tokens, feats):
    """
    Add features indicating how many time a token appears that matches either
    the neg_words or pos_words (defined above). The matching should ignore
    case.

    """
    ###TODO
    feats['neg_words'] = 0
    feats['pos_words'] = 0
    for token in tokens :
        token = token.lower()
        if token in neg_words:
            feats['neg_words'] += 1
        elif token in pos_words:
            feats['pos_words'] += 1
    pass


def cross_validation_accuracy(clf, X, labels, k):
    """
    Compute the average testing accuracy over k folds of cross-validation. You
    can use sklearn's KFold class here (no random seed, and no shuffling
    needed).

    Params:
      clf......A LogisticRegression classifier.
      X........A csr_matrix of features.
      labels...The true labels for each instance in X
      k........The number of cross-validation folds.

    Returns:
      The average testing accuracy of the classifier
      over each fold of cross-validation.
    """
    ###TODO
    cv = KFold(n_splits=k)
    accuracies = []
    for train_ind, tst_ind in cv.split(X):
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[tst_ind])
        accuracies.append(accuracy_score(labels[tst_ind], predictions))
    return np.mean(accuracies)
    pass


def eval_all_combinations(docs, labells, punct_vals,
                          feature_fns, min_freqs):
    """
    Enumerate all possible classifier settings and compute the
    cross validation accuracy for each setting. We will use this
    to determine which setting has the best accuracy.

    For each setting, construct a LogisticRegression classifier
    and compute its cross-validation accuracy for that setting.

    Params:
      docs..........The list of original training documents.
      labels........The true labels for each training document (0 or 1)
      punct_vals....List of possible assignments to
                    keep_internal_punct (e.g., [True, False])
      feature_fns...List of possible feature functions to use
      min_freqs.....List of possible min_freq values to use
                    (e.g., [2,5,10])

    Returns:
      A list of dicts, one per combination. Each dict has
      four keys:
      'punct': True or False, the setting of keep_internal_punct
      'features': The list of functions used to compute features.
      'min_freq': The setting of the min_freq parameter.
      'accuracy': The average cross_validation accuracy for this setting, using 5 folds.
    """
    ###TODO
    res = []
    comb = []
    for i in range(1, len(feature_fns) + 1):
        comb += list(combinations(feature_fns,i))
    for val in punct_vals:
        tokens_list = [tokenize(d, keep_internal_punct=val) for d in docs]
        for feature in comb:
                for min in min_freqs:
                    dic = {}
                    dic['punct:'] = val
                    dic['features:'] = feature
                    dic['min_freq:'] = min
                    X, vocab = vectorize(tokens_list,list(feature), min)
                    dic['accuracy:'] = cross_validation_accuracy(LogisticRegression() , X, labells, 5)
                    res.append(dic)
    return sorted(res, key=lambda x:x['accuracy:'], reverse=True)
    pass
def fit_best_classifier(docs, labels, best_result):
    """
    Using the best setting from eval_all_combinations,
    re-vectorize all the training data and fit a
    LogisticRegression classifier to all training data.
    (i.e., no cross-validation done here)

    Params:
      docs..........List of training document strings.
      labels........The true labels for each training document (0 or 1)
      best_result...Element of eval_all_combinations
                    with highest accuracy
    Returns:
      clf.....A LogisticRegression classifier fit to all
            training data.
      vocab...The dict from feature name to column index.
    """
    ###TODO
    token_lists = [tokenize(doc, keep_internal_punct=best_result['punct:']) for doc in docs]
    X, vocab = vectorize(token_lists, best_result['features:'], best_result['min_freq:'])
    return LogisticRegression().fit(X,labels), vocab
    pass
def parse_test_data(best_result, vocab):
    """
    Using the vocabulary fit to the training data, read
    and vectorize the testing data. Note that vocab should
    be passed to the vectorize function to ensure the feature
    mapping is consistent from training to testing.

    Note: use read_data function defined above to read the
    test data.

    Params:
      best_result...Element of eval_all_combinations
                    with highest accuracy
      vocab.........dict from feature name to column index,
                    built from the training data.
    Returns:
      test_docs.....List of strings, one per testing document,
                    containing the raw.
      test_labels...List of ints, one per testing document,
                    1 for positive, 0 for negative.
      X_test........A csr_matrix representing the features
                    in the test data. Each row is a document,
                    each column is a feature.
    """
    ###TODO
    test_docs, test_labels = read_data('test.csv')
    token_lists = [tokenize(doc, keep_internal_punct=best_result['punct:'])for doc in test_docs ]
    X_test, vocab = vectorize(token_lists, best_result['features:'], best_result['min_freq:'], vocab=vocab)
    return test_docs, test_labels, X_test
def parse_raw_data(best_result, vocab):
    tweets = get_tweets('tweets')
    token_lists = [tokenize(doc, keep_internal_punct=best_result['punct:'])for doc in tweets ]
    X_raw, vocab = vectorize(token_lists, best_result['features:'], best_result['min_freq:'], vocab=vocab)
    return X_raw
def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    """
    Print the n testing documents that are misclassified by the
    largest margin. By using the .predict_proba function of
    LogisticRegression <https://goo.gl/4WXbYA>, we can get the
    predicted probabilities of each class for each instance.
    We will first identify all incorrectly classified documents,
    then sort them in descending order of the predicted probability
    for the incorrect class.
    E.g., if document i is misclassified as positive, we will
    consider the probability of the positive class when sorting.

    Params:
      test_docs.....List of strings, one per test document
      test_labels...Array of true testing labels
      X_test........csr_matrix for test data
      n.............The number of documents to print.

    Returns:
      Nothing; see Log.txt for example printed output.
    """
    ###TODO
    predic_prob = clf.predict_proba(X_test)
    prediction = clf.predict(X_test)

    prob = []
    for ind in np.where(prediction!=test_labels)[0]:
        prob.append((ind, predic_prob[ind][prediction[ind]]))
    for ind in sorted(prob, key=lambda x:-x[1])[:n]:
        print('truth= %d predict=%d proba=%.6f\n' %(test_labels[ind[0]],prediction[ind[0]],ind[1]))
        print(test_docs[ind[0]])
    pass
def classify_results(pos_tweets, neg_tweets):
    """
    add all classes to a dict, each a list
    """
    classify_results = {}
    classify_results['pos'] = pos_tweets
    classify_results['neg'] = neg_tweets
    with open( 'classify_result.pkl', 'wb') as f:
        pickle.dump(classify_results, f)
def predict_result(clf, X_raw):
    predictions = clf.predict(X_raw)
    tweets = get_tweets('tweets')
    neg = []
    pos = []
    for i in range(len(predictions)):
        if predictions[i] == 0:
            neg.append(tweets[i])
        else:
            pos.append(tweets[i])
    classify_results(pos, neg)

def top_coefs(clf, label, n, vocab):
        """
        Find the n features with the highest coefficients in
        this classifier for this label.
        See the .coef_ attribute of LogisticRegression.

        Params:
          clf.....LogisticRegression classifier
          label...1 or 0; if 1, return the top coefficients
                  for the positive class; else for negative.
          n.......The number of coefficients to return.
          vocab...Dict from feature name to column index.
        Returns:
          List of (feature_name, coefficient) tuples, SORTED
          in descending order of the coefficient for the
          given class label.
        """
        ###TODO
        res = []
        if label == 1:
            for ind in np.argsort(clf.coef_[0])[::-1][:n]:
                for k, v in vocab.items():
                    if ind == v:
                        res.append((k, abs(clf.coef_[0][ind])))
        elif label == 0:
            for ind in np.argsort(clf.coef_[0])[:n]:
                for k, v in vocab.items():
                    if ind == v:
                        res.append((k, abs(clf.coef_[0][ind])))
        return res
def find_coef(clf, vocab,  feature_name):
    """ find the coeffience of the specific feature"""
    return clf.coef_[0][vocab[feature_name]]

def main():
    d,l= read_data('train1.csv')
    feature_fns = [token_features, token_pair_features,lexicon_features]

    results = eval_all_combinations(d,l,
                                    [True, False],
                                    feature_fns,
                                    [2, 5, 10])
    # Print information about these results.
    best_result = results[12]
    worst_result = results[-1]

    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    # Fit best classifier.
    clf, vocab = fit_best_classifier(d, l, results[0])
    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' % accuracy_score(test_labels, predictions))
    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 5)
    # Paesr raw data
    X_raw = parse_raw_data(best_result, vocab)
    predict_result(clf,X_raw)
    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t, v) for t, v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t, v) for t, v in top_coefs(clf, 1, 5, vocab)]))


if __name__ == "__main__":
    main()