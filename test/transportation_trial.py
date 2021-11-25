# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 13:16:32 2021

@author: Alberto
"""

from gurobipy import Model, GRB, quicksum
import numpy as np
from matplotlib import pyplot as plt


n = 2 # number of clients
#v = 1 # number of vehicles
costkm = 1 #euro per km

Depot = [(0,0)] #depot
clients = [(1,1),(1,0)]
nodes = Depot + clients
print("nodes:",nodes)

x_nodes = [i[0] for i in nodes]
y_nodes = [i[1] for i in nodes]


# ----- FUNCTIONS -----
def dist(node1,node2):
    return np.sqrt((node1[0]-node2[0])**2+(node1[1]-node2[1])**2)

# ----- MAIN -----
N = [i for i in range(1,n+1)]
V = [0] + N
A = [(i,j) for i in V for j in V if i!=j]
print("links:", A)

distance = [dist(node1,node2) for node1 in nodes for node2 in nodes if node1!=node2]
linkcost = [distance[i]*costkm for i in range(len(A))]
print("cost for each link = ", np.round(linkcost,1))

m = Model('VRP')
x = m.addVars(A, vtype=GRB.BINARY)
u = m.addVars(N, vtype=GRB.CONTINUOUS)


m.addConstrs(quicksum(x[i,j] for j in V if j!=i)==1 for i in N)
m.addConstrs(quicksum(x[i,j] for i in V if i!=j)==1 for j in N)
#m.addConstrs((x[i,j]==1)>> ())

m.modelSense = GRB.MINIMIZE
m.setObjective(quicksum(x[i,j]*linkcost[i] for i, j in A))
m.update()
m.optimize()

activelinks = [a for a in A if x[a].x>0.95]
print("active links",activelinks)

plt.plot(x_nodes[0],y_nodes[0],marker='o')
plt.scatter(x_nodes[1:],y_nodes[1:])

for i,j in activelinks:
    plt.plot([x_nodes[i],x_nodes[j]],[y_nodes[i],y_nodes[j]])