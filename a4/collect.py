"""
Collect data.
"""
import pickle
import sys
import time
from TwitterAPI import TwitterAPI
import random

consumer_key = 'zEPmeBiYKm9EgZ71fe6ayFc0K'
consumer_secret = 'snnt2U9emig7LHV9wQSnJL1C97SMWVEKqbbl3esKzmj2NTWBOs'
access_token = '817424293210394626-gYJwQ5IuNgR1k7X3BY9rmDv5X7r0AST'
access_token_secret = 'r0RzWN25xT3JZN3wLLeKipNjYjuqzJaQCPNZmJNhFhMp8'

def get_twitter():
    """ Construct an instance of TwitterAPI using the tokens you entered above.
    Returns:
      An instance of TwitterAPI.
    """
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)
def read_screen_names(filename):
    """
    Read a text file containing Twitter screen_names, one per line.
    """
    with open(filename) as f:
        lists = f.readlines()
    for i in range(0, len(lists)):
        lists[i]=lists[i].strip('\n')
    return lists

def robust_request(twitter, resource, params, max_tries=5):
    """ If a Twitter request fails, sleep for 15 minutes.
    Do this at most max_tries times before quitting.
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
    """
    users = []
    for i in range(0,len(screen_names)):
        response=robust_request(twitter, 'users/lookup', {'screen_name' : screen_names[i]}, max_tries=5)
        user=[r for r in response]
        users.append(user[0])
    return users
    pass
def get_tweets(twitter, screen_name, num_tweets):
    """
    Retrieve tweets of the user.
    params:
        twiiter......The TwitterAPI object.
        screen_name..The user to collect tweets from.
        num_tweets...The number of tweets to collect.
    returns:
        A list of strings, one per tweet.
    """

    request = robust_request(twitter, 'search/tweets', {'q': screen_name, 'count': num_tweets})
    tweets = [i['text'] for i in request]

    return tweets

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
    """
    for user in users:
        friends = get_friends(twitter, user['screen_name'])
        user['friends']=friends
    pass


def save_obj(obj, name):
    """
    store, list of dicts
    """

    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
def main():
    twitter = get_twitter()
    screen_names = read_screen_names('users.txt')
    print('Established Twitter connection.')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))
    add_all_friends(twitter, users)
    save_obj(users,'users')
    tweets = get_tweets(twitter, screen_names[random.randint(0,len(users)-1)], 100)
    save_obj(tweets,'tweets')
    print(tweets)
if __name__ == "__main__":
    main()