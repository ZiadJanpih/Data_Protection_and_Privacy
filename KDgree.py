import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from random import randrange
import operator
import time


def readGraphFromFile(name, andDraw = False):
    Data = open(name, "r")
    graphType = nx.Graph()
    graph = nx.parse_edgelist(Data, delimiter=',', create_using=graphType, nodetype=int)
    if (andDraw):
        nx.draw(graph)
    return graph

def getNodesDegreesArray(graph):
    degrees = list(graph.degree())
    sortedDegrees = sorted(degrees, key=lambda x:x[1], reverse=True)
    return sortedDegrees    

def Identical(d):
    cost =sum(d[0][1]-d[i][1] for i in range(len(d)))
    I = [(d[i][0],d[0][1]) for i in range(len(d))]
    return cost, I
        
def DPA_GetAnonymizedDegrees(d, k):
    nodesCount = len(d)

    if  nodesCount < 2*k:
        return Identical(d)
    else:
        pairs=[]
        for t in range( k , nodesCount-k+1 , 1):
            DAcost,DA_d =DPA_GetAnonymizedDegrees(d[0:t],k)
            DA_Icost,DA_Id=Identical(d[t:nodesCount])
            All_cost=DAcost+DA_Icost
            All_d=DA_d+DA_Id
            Icost,Id=Identical(d[0:nodesCount])
            if(All_cost >Icost ):
                pairs.append((Icost,Id))
            else:
                pairs.append((All_cost,All_d))
            
        return   min(pairs, key=operator.itemgetter(0))


def DPA_GetAnonymizedDegrees_Optimized(d, k):
    nodesCount = len(d)

    if  nodesCount < 2*k:
        return Identical(d)
    else:
        pairs=[]
        min_t=max(k,(nodesCount - 2*k+1))
        for t in range( min_t , nodesCount-k+1 , 1):
            DAcost,DA_d =DPA_GetAnonymizedDegrees_Optimized(d[0:t],k)
            DA_Icost,DA_Id=Identical(d[t:nodesCount])
            All_cost=DAcost+DA_Icost
            All_d=DA_d+DA_Id
            Icost,Id=Identical(d[0:nodesCount])
            if(All_cost >Icost ):
                pairs.append((Icost,Id))
            else:
                pairs.append((All_cost,All_d))
            
        return   min(pairs, key=operator.itemgetter(0))

def DPA_GetAnonymizedDegrees_Optimized_memorized(d, k,cache=dict()):
    if not cache :
        cache = dict()

    nodesCount = len(d)

    if  nodesCount < 2*k:
        return Identical(d)
    else:
        pairs=[]
        min_t=max(k,(nodesCount - 2*k+1))
        for t in range( min_t , nodesCount-k+1 , 1):
            if (t,nodesCount) in cache.keys():
                record_found = cache.get((t, nodesCount))
                t_cost, anonymized_t = record_found[0], record_found[1]
            else:  
                DAcost,DA_d =DPA_GetAnonymizedDegrees_Optimized_memorized(d[0:t],k,cache)
                DA_Icost,DA_Id=Identical(d[t:nodesCount])
                All_cost=DAcost+DA_Icost
                All_d=DA_d+DA_Id
                Icost,Id=Identical(d[0:nodesCount])
                if(All_cost >Icost ):
                    t_cost,anonymized_t=Icost,Id                   
                else:
                    t_cost,anonymized_t=All_cost,All_d              
                cache[(t, nodesCount)] = (t_cost,anonymized_t)
            pairs.append((t_cost,anonymized_t))
            
        return   min(pairs, key=operator.itemgetter(0))


def calc_I(d):
    I_cost =sum(abs(d[0][1] - e[1])for e in d)
    return I_cost

def calc_Cmerge(d, k):
    m_cost=(d[0][1]-d[k][1])+calc_I(d[k+1:min(k*2,len(d))])
    return m_cost

def calc_Cnew(d, k):
    Cnew = calc_I(d[k: min(k*2,len(d))])
    return Cnew
def GA_GetAnonymizedDegrees(d, k):
    new_group=[(d[i][0],d[0][1]) for i in range(k)]
    merge=True
    index=0
    while(merge):
        if(len(d[index:])<= k):
            rest =[(e[0],d[0][1]) for e in d[k+index:]]
            rrr=new_group+rest
            return rrr
        cMerge = calc_Cmerge(d[index:],k)
        cNew = calc_Cnew(d[index:], k)
        if (cMerge > cNew and len(d[k+index:]) >= k):
            merge=False
            return new_group+GA_GetAnonymizedDegrees(d[k+index:],k)
        else:
            new_group.append((d[k+index][0],d[0][1]))
            index +=1

def getRandomIndex(d):
    positiveValues = d[d>0]
    positiveValueIndex = randrange(np.size(positiveValues))
    return np.argmax(d == positiveValues[positiveValueIndex])

def getVdv(d, nodeIndex, numberOfConnection):
    tempd = np.copy(d)
    vdv = np.empty(0)
    while True:
        index = np.argmax(tempd)
        allIndexes = np.where(d == d[index])[0]
        for i in range(0, np.size(allIndexes)):
            tempd[allIndexes[i]] = 0
            vdv = np.append(vdv, allIndexes[i])
            if (np.size(vdv) == numberOfConnection):
                break
        if (np.size(vdv) == numberOfConnection):
            break; 
    return vdv
    
    
def graphGenerate(d):
    dnp = np.array(list(map(lambda x:x[1], d)))
    newGraph = nx.Graph()
    newGraph.add_nodes_from([d[i][0] for i in range(0,len(d))])
    #print(newGraph.nodes)
    if (np.sum(dnp) % 2 != 0):
        return ('No', None)
    while True:
        if (np.size(dnp[dnp<0]) > 0):
            return ('Unrelaiable', None)
        if (np.size(dnp[dnp == 0]) == np.size(dnp)):
            return ('', newGraph)
        nodeIndex = getRandomIndex(dnp)
        numberOfConnection = dnp[nodeIndex]
        dnp[nodeIndex] = 0
        vdv = getVdv(dnp, nodeIndex, numberOfConnection)
        for i in range(0, np.size(vdv)):
            index = int(vdv[i])
            dnp[index] = dnp[index] - 1
            currentNode = d[nodeIndex][0]
            connectedNode = d[index][0]
            newGraph.add_edge(currentNode, connectedNode)

#Subgraph Generation
def superGraphAlgorithm(graph, k, dNew):
    d = getNodesDegreesArray(graph)
    originalDegrees = np.array(list(map(lambda x:x[1], d)))
    anonymizedDegrees = np.array(list(map(lambda x:x[1], dNew)))
    a = anonymizedDegrees - originalDegrees
    if (np.sum(a) % 2 != 0):
        return ('No', None)
    while True:
        if (np.size(a[a<0]) > 0):
            return ('Unrelaiable', None)
        if (np.size(a[a == 0]) == np.size(a)):
            return ('', graph)
        nodeIndex = getRandomIndex(a)
        numberOfRemainingConnecions = a[nodeIndex]
        a[nodeIndex] = 0
        vdv = getVdv(a, nodeIndex, numberOfRemainingConnecions)
        for i in range(0, np.size(vdv)):
            index = int(vdv[i])
            a[index] = a[index] - 1
            currentNode = dNew[nodeIndex][0]
            connectedNode = dNew[index][0]
            graph.add_edge(currentNode, connectedNode)                

def addNoiseTo(d):
    nodeNumber = len(d)
    noise = np.random.choice([0, 1], size= nodeNumber)
    for i in range(0, nodeNumber):
        if d[i][1] < nodeNumber - 1:
            d[i] = (d[i][0], d[i][1] + noise[i])
        
    sortedD = sorted(d, key=lambda x:x[1], reverse=True)
    return sortedD
    
    
def probingScheme(graph, k):
    d = getNodesDegreesArray(graph)
    dNew = GA_GetAnonymizedDegrees(d, k)
    data = superGraphAlgorithm(graph, k, dNew)
    while data[0] == 'No' or data[0] == 'Unrelaiable':
        d = addNoiseTo(d)
        dNew = GA_GetAnonymizedDegrees(d, k)
        data = superGraphAlgorithm(graph, k, dNew)
    return data[1]

def graphConstruction(graph, k):
    G0 = probingScheme(graph.copy(),k)
    G = Greedy_Swap(graph, G0)
    return G

def Pick20EdgesOf(edgesArr):
    neededEdges = []
    indexes = np.empty(0)
    for i in range(0, 20):
        randomIndex = np.random.randint(0, len(edgesArr), 1)
        while True:
            if randomIndex in indexes[:]:
                randomIndex = np.random.randint(0, len(edgesArr), 1)
            else:
                indexes = np.append(indexes, randomIndex)
                break
    
    for i in range(0, len(indexes)):
        idx = int(indexes[i])
        currentEdge = edgesArr[idx]
        neededEdges.append(currentEdge)
    return neededEdges
    
def FindMaxSwap(graph, newGraph, currentNumberOfIntersection):
    currentDegrees = np.array([val for (node, val) in newGraph.degree()])
    newTemp = newGraph.copy()
    intersectedGraph = nx.algorithms.operators.binary.intersection(graph, newGraph)
    c = len(intersectedGraph.edges)
    allEdges = list(newGraph.edges)
    neededEdges = None
    if (len(allEdges) <= 20):
        neededEdges = allEdges
    else:
        neededEdges = Pick20EdgesOf(allEdges)
    returncurrent = None
    returnSwap = None
    returne1 = None
    returne2 = None
    for i in range(0, len(neededEdges)):
        newTemp = newGraph.copy()
        currentEdge = neededEdges[i]
        for j in  range(i+1, len(neededEdges)):
            newTemp = newGraph.copy()
            swapedEdge = neededEdges[j]
            if (currentEdge[0] != swapedEdge[0]) and (currentEdge[0] != swapedEdge[1]) and (currentEdge[1] != swapedEdge[0]) and (currentEdge[1] != swapedEdge[1]):
                e1 = (currentEdge[0], swapedEdge[0])
                e2 = (currentEdge[1], swapedEdge[1])
                newTemp.remove_edges_from([currentEdge, swapedEdge])
                newTemp.add_edges_from([e1, e2])
                intersectedGraph = nx.algorithms.operators.binary.intersection(graph, newTemp)
                newDegrees = np.sort(np.array([val for (node, val) in newTemp.degree()]))[::-1]
                if (len(intersectedGraph.edges) > c) and (np.array_equal(currentDegrees, newDegrees)):
                    c = len(intersectedGraph.edges)
                    returncurrent = currentEdge
                    returnSwap = swapedEdge
                    returne1 = e1
                    returne2 = e2
                    break
                
                newTemp = newGraph.copy()
                e1 = (currentEdge[0], swapedEdge[1])
                e2 = (currentEdge[1], swapedEdge[0])
                newTemp.remove_edges_from([currentEdge, swapedEdge])
                newTemp.add_edges_from([e1, e2])
                intersectedGraph = nx.algorithms.operators.binary.intersection(graph, newTemp)
                newDegrees = np.sort(np.array([val for (node, val) in newTemp.degree()]))[::-1]
                if (len(intersectedGraph.edges) > c) and (np.array_equal(currentDegrees, newDegrees)):
                    c = len(intersectedGraph.edges)
                    returncurrent = currentEdge
                    returnSwap = swapedEdge
                    returne1 = e1
                    returne2 = e2
                    break
        if returncurrent != None:
            break
    
    return (c, [returncurrent, returnSwap, returne1, returne2])

def Greedy_Swap(originalGraph, G0):
    graph = G0
    sw = FindMaxSwap(originalGraph, graph, 0)
    c = sw[0]
    edges = sw[1]
    newdegrees = [val for (node, val) in graph.degree()]
    while (c > 0):
        if (edges[0] != None):
            graph.remove_edge(*edges[0])
            graph.remove_edge(*edges[1])
            graph.add_edge(*edges[2])
            graph.add_edge(*edges[3])
        sw = FindMaxSwap(originalGraph, graph, 0)
        if (sw[0] <= c):
            break
        else:
            c = sw[0]
            edges = sw[1]
    return graph

def getNodeIndex(nodes, nodeName):
    nodeIndex = np.argmax(nodes == nodeName)
    return nodeIndex

def getVdvPriority(originalGraph, currentGraph, dNew, npDNew, nodeName, nodeIndex, numberOfConnection):
    nodes = np.array(list(map(lambda x: x[0], dNew)))
    tempd = np.copy(npDNew)
    vdv = np.empty(0)
    currentNodeEdges = list(currentGraph.edges(nodeName))
    currentConnectedNodes = list(map(lambda x:x[1], currentNodeEdges))
    
    connectedNodes = np.array(list(map(lambda x: x[1],list(originalGraph.edges(nodeName)))))
    filteredConnectedNodes = list(filter(lambda x: x not in currentConnectedNodes, connectedNodes))
    for i in range(0, np.size(connectedNodes)):
        idx = getNodeIndex(nodes, connectedNodes[i])
        tempd[idx] = 0.0
        if (connectedNodes[i] in filteredConnectedNodes):
            vdv = np.append(vdv, idx)
        
    if np.size(vdv) < numberOfConnection:
        while True:
            index = np.argmax(tempd)
            allIndexes = np.where(npDNew == npDNew[index])[0]
            for i in range(0, np.size(allIndexes)):
                if tempd[allIndexes[i]] != 0.0:
                    tempd[allIndexes[i]] = 0.0
                    vdv = np.append(vdv, allIndexes[i])
                    if (np.size(vdv) == numberOfConnection):
                        break
            if (np.size(vdv) == numberOfConnection):
                break;
    else:
        vdv = vdv[0:numberOfConnection]
    return vdv
    
    
def PriorityAlgorithm(originalGraph, dNew):
    #start_timer = time.time()
    newGraph = nx.Graph()
    newGraph.add_nodes_from([dNew[i][0] for i in range(0,len(dNew))])
    npDNew = np.array(list(map(lambda x:x[1], dNew)))
    if (np.sum(npDNew) % 2 != 0):
        #end_timer = time.time()
        #print('It takes: ', end_timer-start_timer)
        return ('No', None)
    while True:
        if (np.size(npDNew[npDNew<0]) > 0):
            #end_timer = time.time()
            #print('It takes: ', end_timer-start_timer)
            return ('Unrelaiable', None)
        if (np.size(npDNew[npDNew == 0]) == np.size(npDNew)):
            #end_timer = time.time()
            #print('It takes: ', end_timer-start_timer)
            return ('', newGraph)
        nodeIndex = getRandomIndex(npDNew)
        numberOfConnection = npDNew[nodeIndex]
        npDNew[nodeIndex] = 0
        nodeName = dNew[nodeIndex][0]
        vdv = getVdvPriority(originalGraph, newGraph, dNew, npDNew, nodeName, nodeIndex, numberOfConnection)
        for i in range(0, np.size(vdv)):
            index = int(vdv[i])
            npDNew[index] = npDNew[index] - 1
            destNode = dNew[index][0]
            newGraph.add_edge(nodeName, destNode)
            
def checkPriority(graph, k):
    #start_timer = time.time()
    d = getNodesDegreesArray(graph)
    dNew = GA_GetAnonymizedDegrees(d, k)
    data = PriorityAlgorithm(graph, dNew)
    #print('First:', data[0])
    while data[0] == 'No' or data[0] == 'Unrelaiable':
        d = addNoiseTo(d)
        dNew = GA_GetAnonymizedDegrees(d, k)
        data = PriorityAlgorithm(graph, dNew)
        #print('Loop:', data[0])
    #end_timer = time.time()
    #print('It takes: ', end_timer-start_timer)
    return data[1]    


def performance_time(degree_Sequence , k, algorithm):
     start_timer = time.time()
     algorithm(degree_Sequence, k)
     end_timer = time.time()
     return end_timer-start_timer    

def performance_ratio(degree_Sequence , k):

    DPA_cost, DPA_GetAnonymized = DPA_GetAnonymizedDegrees_Optimized_memorized(degree_Sequence, k)

    GA_GetAnonymized=GA_GetAnonymizedDegrees(degree_Sequence, k)
    GA_cost= (sum(abs(GA_GetAnonymized[i][1]-degree_Sequence[i][1])for i in range(len(GA_GetAnonymized))) )
    
    return k,GA_cost/DPA_cost

def Anonymization_cost(algorithm,data,k):
    original=getNodesDegreesArray(data)
    anonymized=None
    if (algorithm == GA_GetAnonymizedDegrees):
       anonymized=GA_GetAnonymizedDegrees(original,k)
    else:
        anonymized=getNodesDegreesArray(algorithm(data,k))
    cost=sum(abs(anonymized[i][1]-original[i][1])for i in range(len(original)))
    return k,cost    