# coding: utf-8

"""
CS579: Assignment 0
Collecting a political social network

In this assignment, I've given you a list of Twitter accounts of 4
U.S. presedential candidates from the previous election.

The goal is to use the Twitter API to construct a social network of these
accounts. We will then use the [networkx](http://networkx.github.io/) library
to plot these links, as well as print some statistics of the resulting graph.

1. Create an account on [twitter.com](http://twitter.com).
2. Generate authentication tokens by following the instructions [here](https://developer.twitter.com/en/docs/basics/authentication/guides/access-tokens.html).
3. Add your tokens to the key/token variables below. (API Key == Consumer Key)
4. Be sure you've installed the Python modules
[networkx](http://networkx.github.io/) and
[TwitterAPI](https://github.com/geduldig/TwitterAPI). Assuming you've already
installed [pip](http://pip.readthedocs.org/en/latest/installing.html), you can
do this with `pip install networkx TwitterAPI`.

OK, now you're ready to start collecting some data!
hods indicated by
#TODO.
I've provided a partial implementation below. Your job is to complete the
code where indicated.  You need to modify the 10 met

Your output should match the sample provided in Log.txt.
"""

# Imports you'll need.
from collections import Counter
import matplotlib.pyplot as plt
from typing import List

import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI

consumer_key = 'zEPmeBiYKm9EgZ71fe6ayFc0K'
consumer_secret = 'snnt2U9emig7LHV9wQSnJL1C97SMWVEKqbbl3esKzmj2NTWBOs'
access_token = '817424293210394626-gYJwQ5IuNgR1k7X3BY9rmDv5X7r0AST'
access_token_secret = 'r0RzWN25xT3JZN3wLLeKipNjYjuqzJaQCPNZmJNhFhMp8'


# This method is done for you.
def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.

    Params:
        filename....Name of the file to read.
    Returns:
        A list of strings, one per screen_name, in the order they are listed
        in the file.

    Here's a doctest to confirm your implementation is correct.
    >>> read_screen_names('candidates.txt')
    ['DrJillStein', 'GovGaryJohnson', 'HillaryClinton', 'realDonaldTrump']
    """
    with open(filename) as f:
        lists = f.readlines()
    for i in range(0, len(lists)):
        lists[i]=lists[i].strip('\n')
    return lists



# I've provided the method below to handle Twitter's rate limiting.
# You should call this method whenever you need to access the Twitter API.
def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
    Args:
      twitter .... A TwitterAPI object.
      resource ... A resource string to request; e.g., "friends/ids"
      params ..... A parameter dict for the request, e.g., to specify
                   parameters like screen_name or count.
      max_tries .. The maximum number of tries to attempt.
    Returns:
      A TwitterResponse object, or None if failed.
    """
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def get_users(twitter, screen_names):
    """Retrieve the Twitter user objects for each screen_name.
    Params:
        twitter........The TwitterAPI object.
        screen_names...A list of strings, one per screen_name
    Returns:
        A list of dicts, one per user, containing all the user information
        (e.g., screen_name, id, location, etc)

    See the API documentation here: https://dev.twitter.com/rest/reference/get/users/lookup

    In this example, I test retrieving two users: twitterapi and twitter.

    >>> twitter = get_twitter()
    >>> users = get_users(twitter, ['twitterapi', 'twitter'])
    >>> [u['id'] for u in users]
    [6253282, 783214]
    """
    users = []
    for i in range(0,len(screen_names)):
        response=robust_request(twitter, 'users/lookup', {'screen_name' : screen_names[i]}, max_tries=5)
        user=[r for r in response]
        users.append(user[0])
    return users
    pass


def get_friends(twitter, screen_name):
    """ Return a list of Twitter IDs for users that this person follows, up to 5000.
    See https://dev.twitter.com/rest/reference/get/friends/ids
    Note, because of rate limits, it's best to test this method for one candidate before trying
    on all candidates.

    Args:
        twitter.......The TwitterAPI object
        screen_name... a string of a Twitter screen name
    Returns:
        A list of ints, one per friend ID, sorted in ascending order.

    Note: If a user follows more than 5000 accounts, we will limit ourselves to
    the first 5000 accounts returned.

    In this test case, I return the first 5 accounts that I follow.
    >>> twitter = get_twitter()
    >>> get_friends(twitter, 'aronwc')[:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    response = robust_request(twitter, 'friends/ids', {'screen_name': screen_name}, max_tries=5)
    friends=[r for r in response]
    friend_ids= sorted(friends)
    return friend_ids
    pass


def add_all_friends(twitter, users):
    """ Get the list of accounts each user follows.
    I.e., call the get_friends method for all 4 candidates.

    Store the result in each user's dict using a new key called 'friends'.

    Args:
        twitter...The TwitterAPI object.
        users.....The list of user dicts.
    Returns:
        Nothing

    >>> twitter = get_twitter()
    >>> users = [{'screen_name': 'aronwc'}]
    >>> add_all_friends(twitter, users)
    >>> users[0]['friends'][:5]
    [695023, 1697081, 8381682, 10204352, 11669522]
    """
    for user in users:
        friends = get_friends(twitter, user['screen_name'])
        user['friends']=friends
    pass


def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    new_users=sorted(users, key=lambda x:x['screen_name'])
    for user in new_users:
        print('%s %d' %(user['screen_name'],len(user['friends'])))
    pass


def count_friends(users):
    """ Count how often each friend is followed.
    Args:
        users: a list of user dicts
    Returns:
        a Counter object mapping each friend to the number of candidates who follow them.
        Counter documentation: https://docs.python.org/dev/library/collections.html#collections.Counter

    In this example, friend '2' is followed by three different users.
    >>> c = count_friends([{'friends': [1,2]}, {'friends': [2,3]}, {'friends': [2,3]}])
    >>> c.most_common()
    [(2, 3), (3, 2), (1, 1)]
    """
    cnt=Counter()
    for user in users:
        for id in user['friends']:
           cnt[id]+=1
    return cnt
    pass


def friend_overlap(users):
    """
    Compute the number of shared accounts followed by each pair of users.

    Args:
        users...The list of user dicts.

    Return: A list of tuples containing (user1, user2, N), where N is the
        number of accounts that both user1 and user2 follow.  This list should
        be sorted in descending order of N. Ties are broken first by user1's
        screen_name, then by user2's screen_name (sorted in ascending
        alphabetical order). See Python's builtin sorted method.

    In this example, users 'a' and 'c' follow the same 3 accounts:
    >>> friend_overlap([
    ...     {'screen_name': 'a', 'friends': ['1', '2', '3']},
    ...     {'screen_name': 'b', 'friends': ['2', '3', '4']},
    ...     {'screen_name': 'c', 'friends': ['1', '2', '3']},
    ...     ])
    [('a', 'c', 3), ('a', 'b', 2), ('b', 'c', 2)]
    """
    overlap=[]
    for i in range(0,len(users)):
        for j in range(i+1, len(users)):
            same_ele=[l for l in users[i]['friends'] if l in users[j]['friends']]
            l=[users[i]['screen_name'],users[j]['screen_name']]
            l.sort()
            overlap.append((l[0], l[1], len(same_ele)))

    tmp=sorted(overlap, key=lambda x:(x[0],x[1]))
    result=sorted(tmp, key=lambda x:x[2], reverse=True)
    return result
    pass


def followed_by_hillary_and_donald(users, twitter):
    """
    Find and return the screen_names of the Twitter users followed by both Hillary
    Clinton and Donald Trump. You will need to use the TwitterAPI to convert
    the Twitter ID to a screen_name. See:
    https://dev.twitter.com/rest/reference/get/users/lookup

    Params:
        users.....The list of user dicts
        twitter...The Twitter API object
    Returns:
        A list of strings containing the Twitter screen_names of the users
        that are followed by both Hillary Clinton and Donald Trump.

    """
    l1=[]
    l2=[]
    for user in users:
       if user['screen_name']=='HillaryClinton':
           l1=user['friends']
       elif user['screen_name']=='realDonaldTrump':
           l2=user['friends']
    common_ids=[x for x in l1 if x in l2]
    common_friends=[]
    for id in common_ids:
        response = robust_request(twitter, 'users/lookup', {'user_id':id}, max_tries=5)
        common_friend = [r for r in response]
        common_friends.append(common_friend[0]['screen_name'])
    return common_friends
    pass


def create_graph(users, friend_counts):
    """ Create a networkx undirected Graph, adding each candidate and friend
        as a node.  Note: while all candidates should be added to the graph,
        only add friends to the graph if they are followed by more than one
        candidate. (This is to reduce clutter.)

        Each candidate in the Graph will be represented by their screen_name,
        while each friend will be represented by their user id.

    Args:
      users...........The list of user dicts.
      friend_counts...The Counter dict mapping each friend to the number of candidates that follow them.
    Returns:
      A networkx Graph
    """
    graph=nx.Graph()
    graph.degree()
    count=0
    for key,value in  friend_counts.items():
        if value >=2:
            graph.add_node(key)
            for user in users:
                graph.add_node(user['screen_name'])
                if key in user['friends']:
                    graph.add_edge(user['screen_name'],key)
    # for i in range(0,len(users)):
    #     for j in range(i+1,len(users)):
    #         if users[j]['id'] in users[i]['friends']:
    #             graph.add_edge(users[i]['screen_name'],users[j]['screen_name'])
    return graph
    pass


def draw_network(graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    pos=nx.spring_layout(graph)
    plt.figure()
    plt.title('Twitter Networking')

    labeldic={}
    for user in users:
        labeldic[user['screen_name']]=user['screen_name']
    nx.draw_networkx(graph, pos, node_size=20, alpha=0.6, font_weight='bold', labels=labeldic, width=0.1,with_labels=True)
    plt.axis('off')
    plt.xticks([])

    plt.yticks([])
    plt.show()
    plt.savefig('network.png')
    pass


def main():
    """ Main method. You should not modify this. """
    twitter = get_twitter()
    screen_names = read_screen_names('candidates.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    print('Friends per candidate:')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))
    print('User followed by Hillary and Donald: %s' % str(followed_by_hillary_and_donald(users, twitter)))

    graph = create_graph(users, friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network(graph, users, 'network.png')
    print('network drawn to network.png')


if __name__ == '__main__':
    main()

# That's it for now! This should give you an introduction to some of the data we'll study in this course.
