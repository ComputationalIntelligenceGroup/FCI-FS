from __future__ import annotations

import warnings
from typing import List, Set, Tuple, Dict
from numpy import ndarray

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Graph import Graph
from causallearn.graph.GeneralGraph import GeneralGraph 
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
from causallearn.utils.DepthChoiceGenerator import DepthChoiceGenerator
from causallearn.utils.cit import *
import causallearn.search.ConstraintBased.FCI as fci

from FAS_FS import fas_fs 



def oneSideElimByPossibleDsep(graph: Graph, independence_test_method: CIT,  node1: Node, node2: Node, edge: Edge,
                              alpha: float, sep_sets: Dict[Tuple[int, int], Set[int]], old_nodes = None ):
    
    def _contains_all(set_a: Set[Node], set_b: Set[Node]):
        for node_b in set_b:
            if not set_a.__contains__(node_b):
                return False
        return True
    
    possibleDsep = fci.getPossibleDsep(node1, node2, graph, -1)
    gen = DepthChoiceGenerator(len(possibleDsep), len(possibleDsep))

    choice = gen.next()
    while choice is not None:
        origin_choice = choice
        choice = gen.next()
        if len(origin_choice) < 2:
            continue
        sepset = tuple([possibleDsep[index] for index in origin_choice])
        #Skip unnecesary independence tests
        if node1 in old_nodes and node2 in old_nodes and all (node in old_nodes for node in sepset):
            continue
        if _contains_all(set(graph.get_adjacent_nodes(node1)), set(sepset)):
            continue
        if _contains_all(set(graph.get_adjacent_nodes(node2)), set(sepset)):
            continue
        X, Y = graph.get_node_map()[node1], graph.get_node_map()[node2]
        condSet_index = tuple([graph.get_node_map()[possibleDsep[index]] for index in origin_choice])
        p_value = independence_test_method(X, Y, condSet_index)
        independent = p_value > alpha
        if independent:
            graph.remove_edge(edge)
            sep_sets[(X, Y)] = set(condSet_index)
            break

def removeByPossibleDsep(graph: Graph, independence_test_method: CIT, alpha: float,
                         sep_sets: Dict[Tuple[int, int], Set[int]], old_nodes = None):
    

    edges = graph.get_graph_edges()
    for edge in edges:
        node_a = edge.get_node1()
        node_b = edge.get_node2()
        
        oneSideElimByPossibleDsep(graph = graph, independence_test_method = independence_test_method,  node1 = node_a, node2 = node_b, 
                                   edge = edge, alpha = alpha, sep_sets = sep_sets, old_nodes = old_nodes )

        if graph.contains_edge(edge):
            
            oneSideElimByPossibleDsep(graph = graph, independence_test_method = independence_test_method,  node1 = node_b, node2 = node_a, 
                                       edge = edge, alpha = alpha, sep_sets = sep_sets, old_nodes = old_nodes )
                



def fci_fs(dataset: ndarray, independence_test_method: str=fisherz, alpha: float = 0.05, 
           initial_sep_sets: Dict[Tuple[int, int], Set[int]] = {}, initial_graph: GeneralGraph = GeneralGraph([]), 
           depth: int = -1, max_path_length: int = -1, verbose: bool = False, new_node_names:List[str] = None, **kwargs) -> Tuple[Graph, List[Edge]]:
    """
    Perform Fast Causal Inference (FCI) algorithm for causal discovery

    Parameters
    ----------
    dataset: data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    independence_test_method: str, name of the function of the independence test being used
            [fisherz, chisq, gsq, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - kci: Kernel-based conditional independence test
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    depth: The depth for the fast adjacency search, or -1 if unlimited
    max_path_length: the maximum length of any discriminating path, or -1 if unlimited.
    verbose: True is verbose output should be printed or logged
   

    Returns
    -------
    graph : a GeneralGraph object, where graph.graph[j,i]=1 and graph.graph[i,j]=-1 indicates  i --> j ,
                    graph.graph[i,j] = graph.graph[j,i] = -1 indicates i --- j,
                    graph.graph[i,j] = graph.graph[j,i] = 1 indicates i <-> j,
                    graph.graph[j,i]=1 and graph.graph[i,j]=2 indicates  i o-> j.
    edges : list
        Contains graph's edges properties.
        If edge.properties have the Property 'nl', then there is no latent confounder. Otherwise,
            there are possibly latent confounders.
        If edge.properties have the Property 'dd', then it is definitely direct. Otherwise,
            it is possibly direct.
        If edge.properties have the Property 'pl', then there are possibly latent confounders. Otherwise,
            there is no latent confounder.
        If edge.properties have the Property 'pd', then it is possibly direct. Otherwise,
            it is definitely direct.
    """

    if dataset.shape[0] < dataset.shape[1]:
        warnings.warn("The number of features is much larger than the sample size!")

    independence_test_method = CIT(dataset, method=independence_test_method, **kwargs)

    ## ------- check parameters ------------
    if (depth is None) or type(depth) != int:
        raise TypeError("'depth' must be 'int' type!")
    if type(max_path_length) != int:
        raise TypeError("'max_path_length' must be 'int' type!")
    ## ------- end check parameters ------------


    old_nodes = initial_graph.get_nodes()
    nodes = []
    
    
   

    # FAS (“Fast Adjacency Search”) is the adjacency search of the PC algorithm, used as a first step for the FCI algorithm.
    graph, sep_sets, test_results = fas_fs(dataset, independence_test_method=independence_test_method, alpha=alpha,
                                         depth=depth, verbose=verbose, new_node_names = new_node_names)
    
    nodes = graph.get_nodes()
    
   
        

    # pdb.set_trace()
    fci.reorientAllWith(graph, Endpoint.CIRCLE)

    fci.rule0(graph, nodes, sep_sets, None, verbose)

    removeByPossibleDsep(graph, independence_test_method, alpha, sep_sets, old_nodes)

    fci.reorientAllWith(graph, Endpoint.CIRCLE)
    fci.rule0(graph, nodes, sep_sets, None, verbose)

    change_flag = True
    

    while change_flag:
        change_flag = False
        change_flag = fci.rulesR1R2cycle(graph, None, change_flag, verbose)
        change_flag = fci.ruleR3(graph, sep_sets, None, change_flag, verbose)

        if change_flag:
            change_flag = fci.ruleR4B(graph, max_path_length, dataset, independence_test_method, alpha, sep_sets,
                                  change_flag,
                                  None, verbose)

            
            if verbose:
                print("Epoch")

        # rule 5
        change_flag = fci.ruleR5(graph, change_flag, verbose)
        
        # rule 6
        change_flag = fci.ruleR6(graph, change_flag, verbose)
        
        # rule 7
        change_flag = fci.ruleR7(graph, change_flag, verbose)
        
        # rule 8
        change_flag = fci.rule8(graph,nodes, change_flag)
        
        # rule 9
        change_flag = fci.rule9(graph, nodes, change_flag)
        # rule 10
        change_flag = fci.rule10(graph, change_flag)

    graph.set_pag(True)

    edges = fci.get_color_edges(graph)

    return graph, edges, sep_sets
