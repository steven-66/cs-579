"""
Summarize data.
"""
import pickle
import random

def load_file(filename):
    """
    load stored data
    """

    with open(filename + '.pkl', 'rb') as f:
        return pickle.load(f)
def ave_num_clusters(clusters):
    total_users = 0
    for cluster in clusters:
        total_users += len(cluster.nodes())
    average = total_users / len(clusters)

    return average
def summarize(filename):
    file = open(filename,'w',encoding='utf-8')
    users = load_file('users')
    file.write("There are %d initial users collected.\n" % (len(users)))
    file.write('\n')
    tweets = load_file('tweets')
    file.write(
        'Collected %d tweets, which is the maximum number of tweets we could collect.\n' % len(
            tweets))
    file.write('\n')
    clusters = load_file('clusters')
    file.write("%d communities are founded\n" % (len(clusters)))
    file.write('\n')
    a_n_clusters = ave_num_clusters(clusters)
    file.write('Average number of users per community: %d\n' % (a_n_clusters))
    file.write('\n')
    classify_results = load_file('classify_result')
    file.write('The positive class has %d instances\n' % (len(classify_results['pos'])))
    file.write('The negative class has %d instances\n' % (len(classify_results['neg'])))
    file.write('\n')
    file.write('Positive example:\n%s\n' % (classify_results['pos'][random.randint(1,10)]))
    file.write('Negative example:\n%s\n' % (classify_results['neg'][random.randint(1,10)].encode('utf-8')))

    print('Mission Complete.')
def main():
    summarize('summary.txt')

if __name__ == "__main__":
    main()