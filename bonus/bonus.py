import networkx as nx

def jaccard_wt(graph, node):
    """
    The weighted jaccard score, defined above.
    Args:
      graph....a networkx graph
      node.....a node to score potential new edges for.
    Returns:
      A list of ((node, ni), score) tuples, representing the
                score assigned to edge (node, ni)
                (note the edge order)
    """
    res =[]
    A = set(graph.neighbors(node))
    for ni in graph.nodes():
        B = set(graph.neighbors(ni))
        nominator = 0
        for i in (A & B):
            nominator += 1/graph.degree(i)
        denominatorA = 0
        for i in A:
            denominatorA += graph.degree(i)
        denominatorA = 1/denominatorA
        denominatorB = 0
        for i in B:
            denominatorB += graph.degree(i)
        denominatorB = 1/denominatorB
        score = nominator / (denominatorA+denominatorB)
        res.append(((node, ni),score))
    pass