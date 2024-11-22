"""
Contains functions to create Bayesian network classifiers with various underlying graph structures:
    BN_tree: A Bayesian network classifier augmented by a tree
    BN_forest: A Bayesian network classifier augmented by a forest
    BN_lines: A Bayesian network classifier augmented by a collection of lines.
"""


from pgmpy.base import DAG
from pgmpy.models.BayesianNetwork import BayesianNetwork
from scipy.stats import truncnorm # Truncated normal distribution
from collections import defaultdict
from decimal import Decimal
import pandas as pd
import networkx as nx
import pylab as plt
import numpy as np

options = {"node_color":"tab:blue",
           "font_color":"whitesmoke",
           "alpha":0.9,
           "font_family":"Arial"}


def BN_tree(d=5, n_children=2, ax=None):
    """
    Bayesian Network with tree structure and with `d` layers. Each node has `n_children` children.
    Inputs:
        d: Depth of tree.
        ax: matplotlib axes. Default=None. If not None, draws plot on axes and displays image
        n_children: Number of children for each node. Default=2 for binary trees.
    Outputs:
        G: A pgmpy DAG representing the tree.
        B: A pgmpy Bayesian Network representing G.
    """
    
    G = DAG()
    total_nodes = int((n_children**d-1)/(n_children-1))+1
    G.add_nodes_from(nodes=list(range(total_nodes)))
    edges = [(0, i) for i in range(1, total_nodes)] # Edges from node 0
    for i in range(1, total_nodes-n_children**(d-1)): # For all nodes not on the bottom layer
        for c in range(n_children): # Attach an edge to each of its children
            edges.append((i, n_children*(i-1)+c+2))

    G.add_edges_from(ebunch=edges)
    B = BayesianNetwork(G)
    
    (G.nodes())
    
    if ax is not None:
        pos = {0: (0,0)}
        for layer in range(d):
            for i in range(n_children**layer):
                width = np.log2(layer+1) # Width of current layer
                relative = (i+1)/(n_children**layer+1) # Position within layer
                pos[len(pos)] = (-width/2 + relative*width, -layer-1)
        edge_color = ["lightgrey" if e[0]==0 else "black" for e in B.edges]
        nx.draw(B, with_labels=True, pos=pos, edge_color=edge_color, ax=ax, **options)
        plt.show()
        
    return G, B


def BN_forest(depths, n_children=2, ax=None, spacing=1):
    """
    Bayesian Network augmented by a forest of trees.
    Inputs:
        depths: An iterable containing the depth of each tree in the forest. (Implicitly defines number of trees)
        n_children: Number of children for each node. Default=2 for binary trees.
        ax: matplotlib axes. Default=None. If not None, draws plot on axes and displays image
        spacing: If drawing forest, gives number of units of spacing between each tree. Default=1.
    Outputs:
        G: A pgmpy DAG representing the forest.
        B: A pgmpy Bayesian Network representing G.
    """
    
    G = DAG()
    G.add_nodes_from(nodes=[0]) # Source node (node 0)
    for d in depths: # For each tree
        nodes_so_far = len(G.nodes())
        new_nodes = list(range(nodes_so_far, nodes_so_far+sum([n_children**x for x in range(d)])))
        G.add_nodes_from(nodes=new_nodes)
        edges = [(0, j) for j in new_nodes] # Edges from node 0
        for i in range(sum([n_children**x for x in range(d-1)])): # For each node not on the bottom layer
            for c in range(n_children): # Attach an edge to each of its children
                edges.append((nodes_so_far+i, nodes_so_far+n_children*i+c+1))
        G.add_edges_from(ebunch=edges)
    B = BayesianNetwork(G)
    
    if ax is not None:
        # `scaling` gives growth rate of layer widths as a function of layer index
        # E.g. can use `x` for identity, `np.log2(x)` for logarithimic, `n_children**x` for exponential.
        scaling = lambda x: x 
        pos = {0: (0,0)}
        tree_widths = [scaling(x)+spacing for x in depths] # Maximum widths reached at the bottom of each tree
        for i, d in enumerate(depths):
            nodes_so_far = len(pos)
            tree_center = (sum(tree_widths[:i])+tree_widths[i]/2) - sum(tree_widths)/2 # Horizontal alignment of tree
            layer = 0 # Track how far down in the tree each node is
            count = 0 # Track the index within the current layer of each node
            for j in range(sum([n_children**x for x in range(d)])):
                if count == n_children**layer:
                    count = 0
                    layer += 1
                count += 1
                width = scaling(layer+1) # Width of current layer
                relative = (count-1)/(n_children**layer-1) if layer!=0 else 0.5 # Relative position within layer (0 to 1)
                pos[nodes_so_far+j] = (tree_center + (-width/2 + relative*width) , -layer-1)
                
        edge_color = ["lightgrey" if e[0]==0 else "black" for e in B.edges]
        nx.draw(B, with_labels=True, pos=pos, edge_color=edge_color, ax=ax, **options)
        plt.show()
        
    return G, B

def BN_lines(lengths, ax=None):
    """
    Bayesian Network augmented by a collection of lines.
    Inputs:
        lengths: An iterable containing the depth of each line in the collection. (Implicitly defines number of lines)
        ax: matplotlib axes. Default=None. If not None, draws plot on axes and displays image.
    Outputs:
        G: A pgmpy DAG representing the network.
        B: A pgmpy Bayesian Network representing G.
    """
    
    G = DAG()
    G.add_nodes_from(nodes=list(range(sum(lengths)+1)))
    edges = [(0, j) for j in range(1,sum(lengths)+1)] # Edges from node 0
    for i,t in enumerate(lengths): # For each line
        edges.extend([(j,j+1) for j in range(sum(lengths[:i])+1,sum(lengths[:i])+t)])
    G.add_edges_from(ebunch=edges)
        
    B = BayesianNetwork(G)
    
    if ax is not None:
        pos = {0: (0,0)}
        for i,t in enumerate(lengths):
            nodes_so_far = sum(lengths[:i])
            for j in range(t):
                pos[nodes_so_far+j+1] = (i-(len(lengths)-1)/2,-j-1)
                
        edge_color = ["lightgrey" if e[0]==0 else "black" for e in B.edges]
        nx.draw(B, with_labels=True, pos=pos, edge_color=edge_color, ax=ax, **options)
        plt.show()
        
    return G, B


def random_extreme(bound, size):
    """
    Helper function for generating probability distributions for Bayesian networks.
    A random array of size `size` with values following Uniform(0,bound) union Uniform(1-bound, 1).
    """
    out = np.random.uniform(0, bound, size=size)
    flips = np.random.randint(0,2, size=size).astype(bool) # Random array of True/False (with equal probability)
    out[flips] = 1 - out[flips] # For each True, flip random number to other side of interval
    return out

def random_disjoint_intervals(intervals, size):
    interval_choices = np.random.choice(range(len(intervals)), size=np.prod(size)) # TODO: Weight intervals
    out = [np.random.uniform(*intervals[i]) for i in interval_choices]
    out = np.asarray(out)
    out = out.reshape(size)
    return out


class BN(BayesianNetwork):
    """Modified BayesianNetwork. Allows for forward pass with given distributions"""

    def __init__(self, G, dists = None, seed = None, mode="interval", bounds=(0.3,0.7)):
        """
        Initialize same as BayesianNetwork. Then create/save distributions.
        """
        self.G = G
        super().__init__(self.G)
        self.edges = self.G.edges()
        self.target_edges = [e for e in self.edges if e[0] == 0] # Edges originating at the target
        self.interior_edges = [e for e in self.edges if e[0] != 0] # Edges not originating at the target
        self.free_nodes = [e[1] for e in self.edges if e[1] not in np.asarray(self.interior_edges)[:,1]] # Nodes that depend only on the source
        self.vforward = np.vectorize(self.forward)
        
        if mode == "interval":
            self.func = lambda size: np.random.uniform(*bounds, size)
        elif mode == "extreme":
            self.func = lambda size: random_extreme(bounds[0], size) # Only 1 bound is needed for random_extreme
        elif mode == "normal" or mode == "gaussian":
            self.func = lambda size: np.clip(np.random.normal(0.5,1, size=size), *bounds) # N(0.5,1), constrained to bounds
        elif mode == "disjoint_intervals":
            self. func = lambda size: random_disjoint_intervals(bounds, size)
        else:
            raise ValueError('`mode` must be one of: "interval", "extreme", or "normal".')

        if dists is not None:
            self.dists = dists
        else:
            np.random.seed(seed)
            # Probabilities for nodes that depend on interior edges.
            # For each node, there are 2^2 possible distributions; 2 for the value of source * 2 for the value of parent.
            self.dists = {e:self.func(size=(2,2)) for e in self.interior_edges}
            # Probabilities for nodes that depend only on source
            self.dists.update({(0,node):self.func(size=2) for node in self.free_nodes})

        self.polynomial = self.calculate_polynomial()

    def forward(self, y=0, seed=None):
        """
        Given a value `y` for the source node, generate values on each node following the distribution of the Bayesian network.
        """
        if seed is not None:
            np.random.seed(seed)
        defined = {0 : y}
        defined.update({node:int(np.random.uniform() < self.dists[(0,node)][y]) for node in self.free_nodes}) # Define all nodes that depend only on 0.
        while len(defined) != len(self.G):
            for e in self.interior_edges:
                if e[0] in defined and e[1] not in defined:
                    defined[e[1]] = int(np.random.uniform() < self.dists[e][defined[e[0]]][y])
        del defined[0] # No need to return y-value
        result = sorted(defined.items(), key=lambda x: x[0]) # Sort results by node index
        result = np.asarray(result)[:,1] # Only return values, not node indices
        return result

    def generate_dataset(self, y, seed=None):
        """
        Given an array `y` of source nodes, calls `self.forward` on each.
        """
        y = np.asarray(y)
        if seed is not None:
            seed = int(1e7 + seed) # We will iterate over numbers [seed, seed+1, seed+2, ...], so we want the seeds to be far apart so that nearby seeds don't generate the same rows
            np.random.seed(seed)
            X = np.asarray([self.forward(row, seed=seed+i) for i, row in enumerate(y)])
        else:
            X = np.asarray([self.forward(row) for row in y])
        X = pd.DataFrame(X, columns=list(range(1,len(self.nodes))))
        return X

    def permutation_probability(self, y, permutation):
        """
        Returns probability of given truth assignment
        """
        prob = Decimal(1.0)
        for node in self.free_nodes:
            p = Decimal(self.dists[(0,node)][y])
            if permutation[node-1] == 0:
                p = 1 - p
            prob *= p
        for e in self.interior_edges:
            p = Decimal(self.dists[e][permutation[e[0]-1]][y])
            if permutation[e[1]-1] == 0:
                p = 1 - p
            prob *= p
        return prob

    def log_odds(self, e, b, a):
        """
        Helper function for self.calculate_polynomial. Shortcut for log(P(...|C=0)/P(...|C=1)).
        where ... is the probability that edge e between x_1 and x_2 has values a and b, respectively (where a, b are 0 or 1)
        """
        numerator = Decimal(self.dists[e][a][1])
        denominator = Decimal(self.dists[e][a][0])
        if b == 0:
            numerator = 1-numerator
            denominator = 1-denominator
        return (numerator/denominator).ln() # Decimal library uses .ln() for log

    def calculate_polynomial(self):
        """
        Return a polynomial representation of the Bayesian network.
        Called on upon BN initialization and output gets saved as self.polynomial
        """
        poly = defaultdict(Decimal)
        const = Decimal(0) # TODO: This should be log(P(C=1)/P(C=0)). Doesn't matter if P(C=1)=P(C=0)=0.5, but does matter otherwise.
        
        for node in self.free_nodes: # Nodes that depend only on the source node
            # x * log(p11/p10) + (1-x) log(p01/p00)
            # self.dists[x,node][c] represents P(X=0 | C=c)
            const += ((1-Decimal(self.dists[0,node][1]))/(1-Decimal(self.dists[0,node][0]))).ln()
            poly[f"x_{node}"] += (Decimal(self.dists[0,node][1])/Decimal(self.dists[0,node][0])).ln()
            poly[f"x_{node}"] -= ((1-Decimal(self.dists[0,node][1]))/(1-Decimal(self.dists[0,node][0]))).ln()
            
        for e in self.interior_edges: # Nodes that depend on some other node as well as the source node
            #   log(p001/p000)*(1-x1)(1-x2) + log(p011/p010)*x1*(1-x2) + log(p101/p100)*(1-x1)*x2 + log(p111/110)*x1*x2
            #     = log(p001/p000) 
            #     + x1 * (-log(p001/p000) + log(p011/p010))
            #     + x2 * (-log(p001/p000) + log(p101/p100))
            #     + x1x2 * (log(p001/p000) - log(p011/p010) - log(p101/p100) + log(p111/110))
            
            # p_abc represents P(X_2=b | X_1=a, C=c)
            # p_a1c is given by self.dists[e][a][c], and p_a0c is given by 1-self.dists[e][a][c]
            
            const += self.log_odds(e, 0, 0)
            poly[f"x_{e[0]}"] += -self.log_odds(e, 0, 0) + self.log_odds(e, 0, 1)
            poly[f"x_{e[1]}"] += -self.log_odds(e, 0, 0) + self.log_odds(e, 1, 0)
            poly[f"x_{e[0]}*x_{e[1]}"] += self.log_odds(e, 0, 0) - self.log_odds(e, 0, 1) - self.log_odds(e, 1, 0) + self.log_odds(e, 1, 1)
        poly["const"] = const
        return poly
    
    def predict_poly(self, permutation):
        """
        Calculate log-odds of a 
        """
        out = 0
        for name, coef in self.polynomial.items():
            if name == "const":
                out += coef
            elif "*" in name: # Interaction terms
                i, j = name.replace("x_", "").split("*")
                out += permutation[int(i)-1]*permutation[int(j)-1]*coef
            else: # Linear terms:
                i = int(name.replace("x_", ""))
                out += permutation[i-1]*coef
        return out