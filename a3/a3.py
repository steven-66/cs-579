# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    tokens = []
    for g in movies['genres']:
        tokens.append(tokenize_string(g))
    movies['tokens'] = tokens
    return movies
    pass
def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    df = defaultdict(lambda :0)

    N = len(movies)
    vocab = {}
    id = 0
    for m in movies['tokens']:
        for i in Counter(m).keys():
            df[i] += 1



    feature = []
    for item in sorted(df.keys()):
        vocab[item] = id
        id += 1
    for m in movies['tokens']:
        tfidf = []
        feat = defaultdict(lambda :0)
        for i in m:
            feat[i] +=1
        max_k = Counter(feat).most_common()[0][1]
        x_row = []
        x_column = []
        for k,v in feat.items():
            tfidf.append(v/max_k * math.log(N/df[k],10))
            x_column.append(vocab[k])
            x_row.append(0)

        X = csr_matrix((tfidf,(x_row,x_column)), shape=(1,len(vocab)))
        feature.append(X)
    movies['feature'] = feature
    return (movies,vocab)
    pass


def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO

    # a = a.toarray()[0]
    # b = b.toarray()[0]
    return np.vdot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

    pass


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    movies_feature = defaultdict()
    for index, row in movies.iterrows():
        movies_feature[row['movieId']] = row['feature']
    usr_ratings = defaultdict(list)
    mean_ratings = defaultdict(lambda : 0)
    for index, row in ratings_train.iterrows():
        usr_ratings[row['userId']].append((movies_feature[row['movieId']],row['rating']))
    predic_ratings = []

    for index,row in ratings_test.iterrows():
        p_rating = 0
        sum_weight = 0
        for item in usr_ratings[row['userId']]:
            weight = cosine_sim(movies_feature[row['movieId']],item[0])
            if weight>0:
                sum_weight += weight
                p_rating += weight*item[1]
        if sum_weight != 0:
            p_rating /= sum_weight
        if p_rating == 0:
            if mean_ratings[row['userId']] == 0:
                for item in usr_ratings[row['userId']]:
                    mean_ratings[row['userId']] += item[1]
                mean_ratings[row['userId']] /= len(usr_ratings[row['userId']])
            p_rating = mean_ratings[row['userId']]
        predic_ratings.append(p_rating)
    return predic_ratings
    pass


def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    # download_data()
    # path = 'ml-latest-small'
    # ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    # movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    # movies = tokenize(movies)
    # movies, vocab = featurize(movies)
    # print('vocab:')
    # print(sorted(vocab.items())[:10])
    # ratings_train, ratings_test = train_test_split(ratings)
    #
    # print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    # predictions = make_predictions(movies, ratings_train, ratings_test)
    # print('error=%f' % mean_absolute_error(predictions, ratings_test))
    # print(predictions[:10])
    # a=[-2.6,0,-0.6,0,0,1.4,0,0,1.4,0,0.4,0]
    a=[1,0,3,0,0,5,0,0,5,0,4,0]
    a = np.asarray(a)
    # b=[-1,1,0,-2,-1,0,0,0,1,0,2,0]
    b=[2,4,0,1,2,0,3,0,4,3,5,0]
    b = np.asarray(b)
    overlap = [i for i in range(len(a)) if a[i]!=0 and b[i]!=0]
    num = ((a[overlap]-3.6)*(b[overlap]-3)).sum()
    print(num)
if __name__ == '__main__':
    main()
