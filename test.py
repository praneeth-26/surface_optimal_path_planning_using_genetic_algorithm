import networkx as nx
from numpy import dot
from numpy.linalg import norm
graph = []
all_path = []

def initPopulation():
    G = nx.Graph()
    with open('Dataset/graph.txt') as file:
        for line in file:
            from_node, to_node, weight = line.strip().split(" ")
            graph.append((from_node, to_node, float(weight)))
            G.add_edge(int(from_node), int(to_node), weight=weight)
        file.close()
    for path in nx.all_simple_paths(G, source=1, target=5):
        all_path.append(path)


def mutation(src,dest):
    weight_value = 0
    for i in range(len(graph)):
        data = graph[i]
        from_node = data[0]
        to_node = data[1]
        weight = data[2]
        if int(src) == int(from_node) and int(dest) == int(to_node):
            weight_value = weight
    return weight_value

def EuclideanDistance(path1,path2):
    point1 = path1
    point2 = path2
    distance = dot(point1, point2)/(norm(point1)*norm(point2))
    return distance
    
def crossOver():
    sp = 1000
    selected = ''
    for i in range(len(all_path)):
        path = all_path[i]
        total_weight = []
        max_weight = []
        w = 0
        for j in range(len(path)-1):
            src = path[j]
            dest = path[j+1]
            weight = mutation(src,dest)
            w = w + weight
            total_weight.append(weight)
            max_weight.append(weight)
        shortest_path = EuclideanDistance(total_weight,max_weight)
        if w < sp and shortest_path  >= 1:
            selected = path
            sp = w
    print(str(selected)+" "+str(sp))        
        

initPopulation()
crossOver()
