# Loading packages that are used in the code
import numpy as np
import pandas as pd
from gurobipy import Model,GRB,LinExpr,quicksum,max_
import matplotlib.pyplot as plt

###################
### MODEL SETUP ###
###################

model = Model()

# Constants
C = 5 # Number of Customers
S = 3 # Number of Shared Delivery Locations
D = 1 # Number of Depots

P = 0.1 # Penalty voor niet thuisbezorgen (delta) 
K = 0.4 # Cost per km


# Locaties
# Alle locaties blijven hetzelfde - anders problemen met branch en bound

Xc = [-0.8,0.2,0.5,-0.3,0.7]
Yc = [0.2,0.5,0.7,-0.3,-0.8]

Xd = 0
Yd = 0

Xs = [-0.5,0.6,0.4]
Ys = [0.8,-0.6,0.4]

Xpos=[0,-0.8,0.2,0.5,-0.3,0.7,-0.5,0.6,0.4]
Ypos=[0,0.2,0.5,0.7,-0.3,-0.8,0.8,-0.6,0.4]


#################
### VARIABLES ###
#################

# Nodes: 0 (Depot) - 1/2/3/4/5 (Customer) - 6/7/8 (SDL)

# X - all links in model
x = {}

## From Depot to customers
# x[0,1] - x[0,5] : from depot to customers
x[0,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X00')
x[0,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X01')
x[0,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X02')
x[0,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X03')
x[0,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X04')
x[0,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X05')
## Depot to SDL
# x[0,6] - x[0,8] : from depot to SDL1/2/3
x[0,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X06')
x[0,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X07')
x[0,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X08')

## Between Customers
# x[1,1] - x[1,5] : from C1 to C1-5
x[1,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X10')
x[1,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X11')
x[1,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X12')
x[1,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X13')
x[1,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X14')
x[1,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X15')
x[1,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X16')
x[1,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X17')
x[1,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X18')
# x[2,1] - x[2,5] : from C2 to C1-5
x[2,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X20')
x[2,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X21')
x[2,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X22')
x[2,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X23')
x[2,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X24')
x[2,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X25')
x[2,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X26')
x[2,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X27')
x[2,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X28')
# x[3,1] - x[3,5] : from C3 to C1-5
x[3,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X30')
x[3,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X31')
x[3,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X32')
x[3,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X33')
x[3,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X34')
x[3,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X35')
x[3,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X36')
x[3,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X37')
x[3,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X38')
# X[4,1]-X[4,5] : from C4 to C1-5
x[4,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X40')
x[4,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X41')
x[4,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X42')
x[4,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X43')
x[4,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X44')
x[4,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X45')
x[4,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X46')
x[4,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X47')
x[4,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X48')
# X[5,1]:X[5,5] from C5 to C1-5
x[5,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X50')
x[5,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X51')
x[5,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X52')
x[5,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X53')
x[5,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X54')
x[5,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X55')
x[5,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X56')
x[5,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X57')
x[5,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X58')

## Between SDL
# x[6,7]-x[7,8] from sdl 6/7/8 to sdl 6/7/8
x[6,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X60')
x[6,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X61')
x[6,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X62')
x[6,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X63')
x[6,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X64')
x[6,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X65')
x[6,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X66')
x[6,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X67')
x[6,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X68')

x[7,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X70')
x[7,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X71')
x[7,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X72')
x[7,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X73')
x[7,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X74')
x[7,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X75')
x[7,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X76')
x[7,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X77')
x[7,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X78')

x[8,0] = model.addVar(lb=0, vtype=GRB.BINARY, name='X80')
x[8,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='X81')
x[8,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X82')
x[8,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X83')
x[8,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X84')
x[8,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X85')
x[8,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='X86')
x[8,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='X87')
x[8,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='X88')

# Y[p,f] - binary varibale indicating whether package p is deliverd to SDL f
y = {}
y[1,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y11')
y[1,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y12')
y[1,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y13')

y[2,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y21')
y[2,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y22')
y[2,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y23')

y[3,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y31')
y[3,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y32')
y[3,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y33')

y[4,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y41')
y[4,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y42')
y[4,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y43')

y[5,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y51')
y[5,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y52')
y[5,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='Y53')

# Z[f] - binary varibale indicating wether SDL f is visited or not
z = {}
z[1] = model.addVar(lb=0, vtype=GRB.BINARY, name='Z1')
z[2] = model.addVar(lb=0, vtype=GRB.BINARY, name='Z2')
z[3] = model.addVar(lb=0, vtype=GRB.BINARY, name='Z3')

# N[i] - binary variable indicating wether customer i is active or not
n = {}
n[1] = model.addVar(lb=0, vtype=GRB.BINARY, name='N1')
n[2] = model.addVar(lb=0, vtype=GRB.BINARY, name='N2')
n[3] = model.addVar(lb=0, vtype=GRB.BINARY, name='N3')
n[4] = model.addVar(lb=0, vtype=GRB.BINARY, name='N4')
n[5] = model.addVar(lb=0, vtype=GRB.BINARY, name='N5')


# Update the model (important!) so all variables are added to the model  
model.update()

# c[i,j] cost from node i to node j 
# c[i,j]=k*d[i,j] - Cost per km times distance (km)

c={}
d={}
for i in range(0,C+S+D,1):
     for j in range(0,C+S+D,1):
          # d[i,j] = 1
          d[i,j] = np.sqrt( (Xpos[j]-Xpos[i])**2 + (Ypos[j]-Ypos[i])**2 )
          c[i,j] = K*d[i,j]


###################
### CONSTRAINTS ###
###################

## 1. Each package must be delivered either to customer or SDL
for j in range(1,C+1,1):
     thisLHS = LinExpr()
     for i in range(0,C+S+D,1): # was range(0,C+S+D,1)
          thisLHS = thisLHS + x[i,j]
     for f in range(1,S+1,1):
          thisLHS = thisLHS + y[j,f]
     model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=1, name='Package %d to C or S' % j)

# 2. Route not reversible x[i,j] + x[j,i] = 1 --> x[i,j] = 1 OR x[j,i] = 1 UNLESS NODE = SDL
for i in range(0,C+D,1): # Voor alle nodes behalve SDLs
     for j in range(0,C+D,1):
          thisLHS = LinExpr()
          thisLHS = thisLHS + x[i,j] + x[j,i]
          model.addConstr(lhs=thisLHS, sense=GRB.LESS_EQUAL, rhs=1, name='Route not reversible for node %d' % j)

## 3. x[i,i] = 0 - They do not exist... 
model.addConstr(lhs=x[0,0], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)
model.addConstr(lhs=x[1,1], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)
model.addConstr(lhs=x[2,2], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)
model.addConstr(lhs=x[3,3], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)
model.addConstr(lhs=x[4,4], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)
model.addConstr(lhs=x[5,5], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)
model.addConstr(lhs=x[6,6], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)
model.addConstr(lhs=x[7,7], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)
model.addConstr(lhs=x[8,8], sense=GRB.EQUAL, rhs=0, name='Continuity %d' % j)

## 4a. Ensure that if there is at least 1 package going to SDL f, then SDL f must also be visited
for f in range(1,S+1,1):
     # thisLHS = LinExpr()
     # thisRHS = LinExpr()
     # for p in range(1,C+1,1):
     #      thisLHS = thisLHS+y[p,f]
     # thisLHS = np.ceil(thisLHS/C)
     # thisRHS = thisRHS + z[f]           
     # model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=thisRHS, name='SDL %d visited if package goes to SDL' % f)
     model.addConstr(z[f] == max_(y[1,f],y[2,f],y[3,f],y[4,f],y[5,f]))

## 4b. Ensure that if SDL f is visited, there must be at least 1 route going to this SDL
for f in range(1,S+1,1):
     thisLHS = LinExpr()
     thisRHS = LinExpr()
     for i in range(0,C+S+D,1):
          thisRHS = thisRHS + x[i,C+f]
     thisLHS = thisLHS + z[f]
     model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=thisRHS, name='Route to SDL %d if visited' % f)
     
## 6. There must start a route at the depot.
thisLHS = LinExpr()
for j in range(0,C+S+D,1):
     thisLHS = thisLHS + x[0,j]
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=1, name='Starting route from depot')    

## 7. There may be only one departure for each node.
for i in range(0,C+S+D,1):
     thisLHS = LinExpr()
     for j in range(0,C+S+D,1):
          thisLHS = thisLHS + x[i,j]
     model.addConstr(lhs=thisLHS, sense=GRB.LESS_EQUAL, rhs=1, name='One departure per node')

## 8. There must be an arrival at the depot.
thisLHS = LinExpr()
for j in range(0,C+S+D,1): # Exception for the depot
     thisLHS = thisLHS + x[j,0]
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=1, name='Arrival for the depot')

## 9. If customer i is active - Ni =1
for i in range(1,C+1,1):
     thisLHS = LinExpr()
     for f in range(1,S+1,1):
          thisLHS = thisLHS + y[i,f]
     thisLHS = thisLHS + n[i]
     model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=1, name='Ni is 1 if customer %d (i) is active' % i)

## 10. # Active Links = # Active Nodes
thisLHS = LinExpr()
thisRHS = LinExpr()
for i in range(0,C+S+D,1): # number of active links
     for j in range(0,C+S+D,1):
          thisLHS = thisLHS + x[i,j]
for f in range(1,S+1,1): # number of active sdls
     thisRHS = thisRHS + z[f]
for i in range(1,C+1,1): # number of active customers
     thisRHS = thisRHS + n[i]
thisRHS = thisRHS + 1 # number of active depots (always, 1)
model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=thisRHS, name='Active Links = Active Nodes')

## 11. If there is an arriving route at a node, there must also be a leaving route
for j in range(0,C+S+D,1):
     thisLHS = LinExpr()
     thisRHS = LinExpr()
     for i in range(0,C+S+D,1):
          thisLHS = thisLHS + x[i,j]
          thisRHS = thisRHS + x[j,i]
     model.addConstr(lhs=thisLHS, sense=GRB.EQUAL, rhs=thisRHS, name='If arrving, then also leaving route at node %d' % j)

## 12. Subtour elimination C2-C3-S3
thisLHS = LinExpr()
thisLHS = thisLHS + x[2,3]+x[3,8]+x[8,2]
model.addConstr(lhs = thisLHS, sense=GRB.LESS_EQUAL, rhs=2, name='Subtour elimination C2->C3->S3')

thisLHS = LinExpr()
thisLHS = thisLHS + x[3,2]+x[2,8]+x[8,3]
model.addConstr(lhs = thisLHS, sense=GRB.LESS_EQUAL, rhs=2, name='Subtour elimination C3->C2->S3')

## 13. Subtour elimination C5-S2
thisLHS = LinExpr()
thisLHS = x[5,7]+x[7,5]
model.addConstr(lhs=thisLHS, sense=GRB.LESS_EQUAL, rhs=1, name='Subtour elimination C5<->S2')


## 14. Subtour elimination C2-S3
thisLHS = LinExpr()
thisLHS = x[2,8]+x[8,2]
model.addConstr(lhs=thisLHS, sense=GRB.LESS_EQUAL, rhs=1, name='Subtour elimination C2<->S3')

model.update()

#################
### OBJECTIVE ###
#################

# Defining objective function     
obj = LinExpr() 

for i in range(0,C+S+D,1):
     for j in range(0,C+S+D,1):
          if j!=i:
               obj = obj + 0.5*c[i,j]*x[i,j] # x[i,j] loopt van x[0,1] tot x[4,5]

for p in range(1,C+1,1):
     for f in range(1,S+1,1):
          obj = obj + P*y[p,f]

# Important: here we are telling the solver we want to minimize the objective
# function. Make sure you are selecting the right option!    
model.setObjective(obj,GRB.MINIMIZE)
# Updating the model
model.update()
# Writing the .lp file. Important for debugging
model.write('model_formulation.lp')  

####################
### OPTIMIZATION ###
####################
model.display()

# Here the model is actually being optimized
model.optimize()

# Saving our solution in the form [name of variable, value of variable]
solution = []
nzsolution = []
for v in model.getVars():
     solution.append([v.varName,v.x])
     if v.x != 0.0: # Print all nonzero variables and its values
          print(v.varName, v.x)
          nzsolution.append([v.varName, v.x])

# Plot all nodes (C, SDL, D)
plt.figure(1)
plt.plot(Xc,Yc,'o',linewidth=2)
plt.plot(Xd,Yd,'x',linewidth=2)
plt.plot(Xs,Ys,'*',linewidth=2)
plt.xlim((-1,1))
plt.ylim((-1,1))
plt.legend(['Customer','Depot','SDL'])
plt.title('Low Penalty, High Cost', fontsize='medium')
plt.suptitle('Solution')

## Plot all active links
nzlinks = [] # Get all links
for sol in range(np.shape(nzsolution)[0]):
     nzlinks.append([nzsolution[sol][0]])

nzx = [] # Get all links containing 'X'
for lnk in range(np.shape(nzlinks)[0]):
     if "X" in nzlinks[lnk][0]:
          nzx.append(nzlinks[lnk][0])

for idx in range(np.size(nzx)):# Plot all active links containing 'X'
     from_idx = int(nzx[idx][1])
     to_idx = int(nzx[idx][2])
     # Plot link from node(from_idx) to node(to_idx)
     plt.plot([Xpos[from_idx],Xpos[to_idx]],[Ypos[from_idx],Ypos[to_idx]],'-k',linewidth=1)

# Add labels to the plot
lbl = ['D','C1','C2','C3','C4','C5','S1','S2','S3']
for k in range(np.size(Xpos)):
     plt.text(Xpos[k]+0.05,Ypos[k],lbl[k],verticalalignment='center_baseline',horizontalalignment='left')
plt.show()