
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np, itertools
from tkinter import simpledialog
from tkinter import filedialog
from collections import deque
import networkx as nx
from scipy.spatial import Delaunay
from numpy import dot
from numpy.linalg import norm

INFINITY = float("inf")

main = tkinter.Tk()
main.title("Surface Optimal Path Planning Using an Extended Dijkstra Algorithm") #designing main screen
main.geometry("1300x1200")

global existing_diskstra_length, extended_diskstra_length
graph = []
global nodesList
global adjacencyList

def upload():
    global filename
    global tf1
    global nodesList
    global adjacencyList
    graph.clear()
    
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    tf1.insert(END,filename)

    with open(filename) as file:
        for line in file:
            from_node, to_node, weight = line.strip().split(" ")
            graph.append((from_node, to_node, float(weight)))
    file.close()
    nodesList = set()
    for nodeedge in graph:
        nodesList.update([nodeedge[0], nodeedge[1]])
    adjacencyList = {node: set() for node in nodesList}
    for edges in graph:
        adjacencyList[edges[0]].add((edges[1], edges[2]))

def generateGraph():
    G = nx.Graph()
    temp = {}
    for i in range(len(graph)):
        data = graph[i]
        from_node = data[0]
        to_node = data[1]
        weight = data[2]
        temp[(from_node,to_node)] = weight
        G.add_edge(from_node, to_node, weight=weight)
    
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=temp)
    plt.show()
    
def traditionalDijkstra():
    global existing_diskstra_length
    text.delete('1.0', END)
    start_node = tf2.get()
    end_node = tf3.get()
    unvisited_nodes = nodesList.copy()
    distance_from_start = {
            node: (0 if node == start_node else INFINITY) for node in nodesList
        }
    previous_node = {node: None for node in nodesList}
    while unvisited_nodes:
        current_node = min(
            unvisited_nodes, key=lambda node: distance_from_start[node]
            )
        unvisited_nodes.remove(current_node)
        if distance_from_start[current_node] == INFINITY:
            break
        for neighbor, distance in adjacencyList[current_node]:
            new_path = distance_from_start[current_node] + distance
            if new_path < distance_from_start[neighbor]:
                distance_from_start[neighbor] = new_path
                previous_node[neighbor] = current_node
            if current_node == end_node:
                break 

    path = deque()
    current_node = end_node
    while previous_node[current_node] is not None:
        path.appendleft(current_node)
        current_node = previous_node[current_node]
    path.appendleft(start_node)
    existing_diskstra_length = distance_from_start[end_node]
    text.insert(END,'\ngraph definition file: {0}'.format(filename)+"\n")
    text.insert(END,'      start/end nodes: {0} -> {1}'.format(start_node, end_node)+"\n")
    text.insert(END,'        shortest path: {0}'.format(path)+"\n")
    text.insert(END,'       Traditional Dijkstra total path length: {0}'.format(existing_diskstra_length)+"\n")
        

def initPopulation(start_node,end_node):#function to initialize population from dataset
    graph = []
    all_path = []
    G = nx.Graph()
    with open('Dataset/graph.txt') as file:
        for line in file:
            from_node, to_node, weight = line.strip().split(" ")
            graph.append((from_node, to_node, float(weight)))
            G.add_edge(int(from_node), int(to_node), weight=weight)
        file.close()
    for path in nx.all_simple_paths(G, source=start_node, target=end_node):
        all_path.append(path)
    return graph, all_path

def mutation(src,dest,graph):#mutation function
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

def crossOver(graph, all_path,start_node,end_node):#crossover function
    sp = 1000
    selected = ''
    global extended_diskstra_length
    for i in range(len(all_path)):
        path = all_path[i]
        total_weight = []
        max_weight = []
        w = 0
        for j in range(len(path)-1):
            src = path[j]
            dest = path[j+1]
            weight = mutation(src,dest,graph) #perform mutation
            w = w + weight
            total_weight.append(weight)
            max_weight.append(weight)
        shortest_path = EuclideanDistance(total_weight,max_weight)#compute fitness using euclidean distance
        if w < sp and shortest_path  >= 1:
            selected = path
            sp = w
    print(str(selected)+" "+str(sp))
    text.insert(END,'\ngraph definition file: {0}'.format(filename)+"\n")
    text.insert(END,'      start/end nodes: {0} -> {1}'.format(start_node, end_node)+"\n")
    text.insert(END,'        shortest path: {0}'.format(selected)+"\n")
    text.insert(END,'       Traditional Dijkstra total path length: {0}'.format(sp)+"\n")
    extended_diskstra_length = sp

def geneticAlgorithm():
    global extended_diskstra_length
    start_node = tf2.get()
    end_node = tf3.get()
    graph, all_path = initPopulation(int(start_node),int(end_node))#initializing population from dataset
    crossOver(graph, all_path,int(start_node),int(end_node))#calling crossover function for mutation
    


def plotGraph():
    height = [existing_diskstra_length,extended_diskstra_length]
    bars = ('Existing Dijkstra Path Length', 'Extended Dijkstra Path Length')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Surface Optimal Path Planning Using an Extended Dijkstra Algorithm')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=300)
text.config(font=font1)


font1 = ('times', 12, 'bold')
l1 = Label(main, text='Upload Dataset')
l1.config(font=font1)
l1.place(x=50,y=100)

tf1 = Entry(main,width=40)
tf1.config(font=font1)
tf1.place(x=200,y=100)

uploadButton = Button(main, text="Browse", command=upload)
uploadButton.place(x=550,y=100)
uploadButton.config(font=font1)

l2 = Label(main, text='Source')
l2.config(font=font1)
l2.place(x=50,y=150)

tf2 = Entry(main,width=10)
tf2.config(font=font1)
tf2.place(x=130,y=150)

l3 = Label(main, text='Destination')
l3.config(font=font1)
l3.place(x=240,y=150)

tf3 = Entry(main,width=10)
tf3.config(font=font1)
tf3.place(x=350,y=150)


generateButton = Button(main, text="Generate Graph", command=generateGraph)
generateButton.place(x=50,y=200)
generateButton.config(font=font1)

traditionalButton = Button(main, text="Run Traditional Dijkstra Algorithm", command=traditionalDijkstra)
traditionalButton.place(x=330,y=200)
traditionalButton.config(font=font1)

extendedButton = Button(main, text="Run Genetic Algorithm", command=geneticAlgorithm)
extendedButton.place(x=50,y=250)
extendedButton.config(font=font1)

graphButton = Button(main, text="Path Length Comparison Graph", command=plotGraph)
graphButton.place(x=330,y=250)
graphButton.config(font=font1)

main.config(bg='OliveDrab2')
main.mainloop()
