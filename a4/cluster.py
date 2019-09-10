"""
Cluster data.
"""
from collections import Counter,defaultdict,deque
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import math
def get_user_info(name):
    """
    load stored user info, list of dicts
    (screen_name, id, friends_id)
    """

    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
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
    count=0
    for key,value in  friend_counts.items():
        if value >=2:
            graph.add_node(key)
            for user in users:
                graph.add_node(user['screen_name'])
                if key in user['friends']:
                    graph.add_edge(user['screen_name'],key)
    return graph
    pass


def draw_network(title,graph, users, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    pos=nx.spring_layout(graph)
    plt.figure()
    plt.figure(figsize=(10, 10))
    plt.title(title)
    labeldic={}
    for user in users:
        labeldic[user['screen_name']]=user['screen_name']
    nx.draw_networkx(graph,pos=pos,node_size=100, alpha=0.6, font_weight='bold', labels=labeldic, width=0.5,with_labels=True)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(filename)
    pass
def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. To reduce running time, the max_depth parameter ends
    the search after the specified depth.
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node to this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    """
    q=deque()
    q.append(root)
    seen=set()
    discovered=set()
    node2distances=defaultdict(int)
    node2num_paths=defaultdict(int)
    node2num_paths[root]=1
    node2parents=defaultdict(list)
    while len(q)>0:
        n=q.popleft()
        if n not in seen:
            seen.add(n)
        for nn in graph.neighbors(n):
            if nn not in seen:
                if node2distances[n]+1 <= max_depth and node2distances[n]!= node2distances[nn] or node2distances[n]==0:
                    if nn not in discovered:
                        discovered.add(nn)
                        q.append(nn)
                        node2distances[nn] = node2distances[n]+1
                        node2num_paths[nn] = 1
                        node2parents[nn].append(n)
                    else:
                        if node2distances[nn] == node2distances[n] + 1:
                            node2num_paths[nn] = node2num_paths[nn] + 1
                            node2parents[nn].append(n)
    return node2distances, node2num_paths, node2parents
    pass

def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. Each key is a tuple of two strings
      representing an edge (e.g., ('A', 'B')). Make sure each of these tuples
      are sorted alphabetically (so, it's ('A', 'B'), not ('B', 'A')).
    """
    ###TODO
    credits=defaultdict(lambda :1.0)
    res=defaultdict(int)
    for node,distance in sorted(node2distances.items(), key=lambda x:x[1], reverse=True):
            for p in node2parents[node]:
                # if tuple(sorted([str(p), str(node)])) == ('15846407','jk_rowling'):
                #     print(tuple(sorted([str(p), str(node)])))

                credits[p] += credits[node]/node2num_paths[node]*node2num_paths[p]
                res[tuple(sorted([str(p),str(node)]))] = credits[node]/node2num_paths[node]*node2num_paths[p]
    return res
    pass


def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    You should call the bfs and bottom_up functions defined above for each node
    in the graph, and sum together the results. Be sure to divide by 2 at the
    end to get the final betweenness.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.
    """
    result = defaultdict(int)
    for node in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, node, max_depth)
        for k,v in bottom_up(node, node2distances, node2num_paths, node2parents).items():
            result[k] += v
    for k in result.keys():
        result[k] /=2
    return result
    pass


def get_components(graph):
    """
    A helper function you may use below.
    Returns the list of all connected components in the given graph.
    """
    return [c for c in nx.connected_component_subgraphs(graph)]

def partition_girvan_newman(graph, max_depth, nums ):

    betweenness = sorted(approximate_betweenness(graph,max_depth).items(), key=lambda x:x[1], reverse=True)
    components=get_components(graph)
    while len(components) < nums:
        remove_edge = betweenness[0][0]
        rm = (int(remove_edge[0]),remove_edge[1]) # convert friends' id into integer
        graph.remove_edge(*rm)
        components = get_components(graph)
        betweenness.pop(0)
    clusters = [com for com in components if len(com)>1]
    return clusters,graph
    pass

def get_subgraph(graph, min_degree):
    """Return a subgraph containing node whose degree is
    greater than or equal to min_degree.
    """
    results=set()
    for node in graph.nodes():
        if graph.degree(node)>=min_degree:
            results.add(node)
    return graph.subgraph(results)
    pass
def save_obj(obj, name):
    """
    store, list of dicts
    """

    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def main():
    print('Friends per candidate:')
    users = get_user_info('users')
    print_num_friends(users)
    friend_counts = count_friends(users)
    print('Most common friends:\n%s' % str(friend_counts.most_common(5)))
    print('Friend Overlap:\n%s' % str(friend_overlap(users)))

    graph = create_graph(users,friend_counts)
    print('graph has %s nodes and %s edges' % (len(graph.nodes()), len(graph.edges())))
    draw_network('Twitter Networking',graph, users, 'original_network.png')
    cluster,graph = partition_girvan_newman(graph,10,100)
    cluster = sorted(cluster, key =lambda x:len(x.nodes), reverse=True)
    save_obj(cluster,'clusters')
    # print('cluster 1 has %d nodes, cluster 2 has %d nodes, cluster 3 has %d nodes' %
    #       (len(cluster[0].nodes()), len(cluster[1].nodes()), len(cluster[2].nodes())))
    draw_network('Twitter Community',graph,users, 'community.png')
if __name__ == "__main__":
    main()