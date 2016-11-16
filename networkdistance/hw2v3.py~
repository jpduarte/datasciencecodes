#https://www.youtube.com/watch?v=gdmfOwyQlcI

from collections import defaultdict, deque
from heapq import *
import numpy as np
import random
import operator

class Graph(object):
    def __init__(self):
        self.nodes = set()
        self.edges = defaultdict(list)
        self.distances = {}

    def add_node(self, value):
        self.nodes.add(value)

    def add_edge(self, from_node, to_node, distance):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.distances[(from_node, to_node)] = distance

def getlowestdistancenode(distances,unvisited):
  i=0
  closest_node = ''
  #start getting maximun value of the distances
  #v=max(distances.values())
  lowesdistance =  0
  for key in unvisited:
    #first one when not distance has been recorded
    if ((distances[key]>0) and (lowesdistance==0)):
      lowesdistance = distances[key]
      closest_node = key  
  
    if ((distances[key]>0) and (distances[key]<lowesdistance)):
      lowesdistance = distances[key]
      closest_node = key
  return closest_node 
  
def dijkstra(from_point,distances,graph):
  distance_accumulated = distances[from_point]
  for neighborn in graph.edges[from_point]:
    print (neighborn)
    
    #first check if edge it was writen a,b or b,a, and assign axuliar distance
    key = (neighborn,from_point)
    if key in graph.distances:
      distance_aux = graph.distances[neighborn,from_point]
    else:
      distance_aux= graph.distances[from_point,neighborn]
    #once distance is calculated first check if distance to that node is negative, if its, first time visited , if not visited, check if distance calculated is larger or lower
    if distances[neighborn]<0:
      distances[neighborn] = distance_aux+distance_accumulated    
    else:
      #only assing if its lower, otherwise, ignore
      if ((distance_aux+distance_accumulated )<distances[neighborn]):
        distances[neighborn] = distance_aux+distance_accumulated
  return distances
########################################################################code runs
#class definitions
graphaux = Graph()
    
#create N random points
N=4
randPoints = np.random.random((N,2))
i=0
#add nodes
for point in randPoints:
    graphaux.add_node(str(i))
    i=i+1

#assing N/2 origin
originPoints = randPoints[0:N/2,:]

#assign N/2 to destination
destinationPoints = randPoints[N/2:,:]

print(randPoints)
print(originPoints)
print(destinationPoints)

#adding and calculating all edges
i=0
for point_a in originPoints:
    j=int(N/2)
    for point_b in destinationPoints:
        graphaux.add_edge(str(i),str(j),np.linalg.norm(point_a-point_b))
        j=j+1
    i=i+1  
    
#checking results    
print ("nodes")
print(graphaux.nodes)
print ("edges")
print(graphaux.edges)
print ("distances")
print(graphaux.distances)

        
initail_nodel = '0'
unvisited = set(graphaux.nodes)
visited = set()
distances = {}

#initialize distances
for node in graphaux.nodes:
  distances[node] =-1
  
#first visit
distances[initail_nodel]=0

################################################################check first distances to initial neighborns
from_point = initail_nodel
print("from point:")
print(from_point)
unvisited.remove(from_point)
visited.add(from_point)

distances = dijkstra(from_point,distances,graphaux)

print("final print:")
print (distances)
print (unvisited)
print (visited)      
#############################################################next visiting round 
from_point = getlowestdistancenode(distances,unvisited)
print("from point:")
print(from_point)
unvisited.remove(from_point)
visited.add(from_point)

distances = dijkstra(from_point,distances,graphaux)

print("final print:")
print (distances)
print (unvisited)
print (visited)  
#############################################################next visiting round 
from_point = getlowestdistancenode(distances,unvisited)
print("from point:")
print(from_point)
unvisited.remove(from_point)
visited.add(from_point)

distances = dijkstra(from_point,distances,graphaux)

print("final print:")
print (distances)
print (unvisited)
print (visited)

#############################################################next visiting round 
from_point = getlowestdistancenode(distances,unvisited)
print("from point:")
print(from_point)
unvisited.remove(from_point)
visited.add(from_point)

distances = dijkstra(from_point,distances,graphaux)

print("final print:")
print (distances)
print (unvisited)
print (visited)

"""
weighted graph:
tentative distance value to every node, 0 to initial one and infitinite to the rest
start with visiting set and unvisited set (rest of node)
start initial node, and remove it from the unvisited set,
calculate the distance to node and from those

start at node, 


"""
