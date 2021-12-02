# Loading packages that are used in the code
import numpy as np
import pandas as pd
from gurobipy import Model,GRB,LinExpr,quicksum
import matplotlib.pyplot as plt

###################
### MODEL SETUP ###
###################

model = Model()

# Constants
C = 5 # Number of Customers
S = 3 # Number of Shared Delivery Locations
D = 1 # Number of Depots

P = 0.2 # Penalty voor thuisbezorgen (delta)
K = 1 # Cost per km


# Locaties
# Alle locaties blijven hetzelfde - anders problemen met branch en bound

Xc = [-0.8,0.2,0.5,-0.3,0.7]
Yc = [0.2,0.5,0.7,-0.3,-0.8]

Xd = 0
Yd = 0

Xs = [-0.5,0.6,0.4]
Ys = [0.8,-0.6,0.4]

Xpos=[]
Xpos = Xpos.append[Xd]
Xpos = Xpos.append[Xc]
Xpos = Xpos.append[Xs]

Ypos=[]
Ypos = Ypos.append[Yd]
Ypos = Ypos.append[Yc]
Ypos = Ypos.append[Ys]


plt.figure(1)
plt.plot(Xc,Yc,'o')
plt.plot(Xd,Yd,'x')
plt.plot(Xs,Ys,'*')
plt.xlim((-1,1))
plt.ylim((-1,1))
plt.legend(['Customer','Depot','SDL'])
plt.title('Nodes')
plt.show()

#################
### VARIABLES ###
#################

# Nodes: 0 (Depot) - 1/2/3/4/5 (Customer) - 6/7/8 (SDL)

# X - all links in model
x = {}

## From Depot to customers
# x[0,1] - x[0,5] : from depot to customers
x[0,1] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xd1')
x[0,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xd2')
x[0,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xd3')
x[0,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xd4')
x[0,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xd5')

## Between Customers
# x[1,2] - x[1,5] : from C1 to C2/3/4/5
x[1,2] = model.addVar(lb=0, vtype=GRB.BINARY, name='X12')
x[1,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X13')
x[1,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X14')
x[1,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X15')
# x[2,3] - x[2,5] : from C2 to C3/4/5
x[2,3] = model.addVar(lb=0, vtype=GRB.BINARY, name='X23')
x[2,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X24')
x[2,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X25')
# x[3,4] - x[3,5] : from C3 to C4/5
x[3,4] = model.addVar(lb=0, vtype=GRB.BINARY, name='X34')
x[3,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X35')
# X[4,5] : from C4 to C5
x[4,5] = model.addVar(lb=0, vtype=GRB.BINARY, name='X45')

## Depot to SDL
# x[0,6] - x[0,8] : from depot to SDL1/2/3
x[0,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xds1')
x[0,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xds2')
x[0,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xds3')

## From Customer to SDL
# x[1,6]-x[1,8]: Cust 1 to SDL 6/7/8
x[1,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc1s1')
x[1,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc1s2')
x[1,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc1s3')
# x[2,6]-x[2,8]: Cust 2 to SDL 6/7/8
x[2,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc2s1')
x[2,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc2s2')
x[2,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc2s3')
# x[3,6]-x[3,8]: Cust 3 to SDL 6/7/8
x[3,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc3s1')
x[3,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc3s2')
x[3,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc3s3')
# x[4,6]-x[4,8]: Cust 4 to SDL 6/7/8
x[4,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc4s1')
x[4,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc4s2')
x[4,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc4s3')
# x[5,6]-x[5,8]: Cust 5 to SDL 6/7/8
x[5,6] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc5s1')
x[5,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc5s2')
x[5,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xc5s3')

## Between SDL
# x[6,7]-x[7,8] from sdl 6/7/8 to sdl 6/7/8
x[6,7] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xs6s7')
x[6,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xs6s8')
x[7,8] = model.addVar(lb=0, vtype=GRB.BINARY, name='Xs7s8')

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


# Update the model (important!) so all variables are added to the model  
model.update()

# c[i,j] cost from node i to node j 
# c[i,j]=k*d[i,j] - Cost per km times distance (km)

c={}
d={}
for i in range(1,C+S+D+1,1):
     for j in range(1,C+S+D+1,1):
          d[i,j] = np.sqrt( (Xpos(j)-Xpos(i))^2 + (Ypos(j)-Ypos(i))^2 )
          c[i,j] = K*d[i,j]




###################
### CONSTRAINTS ###
###################

thisLHS = LinExpr()
# each order must be delivered either to C or to S
for j in range(1,9,1): # for j 1-2-3-4-5-6-7-8
     for i in range(C+S+D+1):
       thisLHS = thisLHS + x[i,j]
     
     



for i in C:        
    for j in S:
        thisLHS = thisLHS + y[i,j]

model.addConstr(lhs=thisLHS, vtype=GRB.BINARY, rhs= 1, name='Const1')

# thisLHS = LinExpr()
# thisLHS += y[9]-y[2]+x[2]
# model.addConstr(lhs=thisLHS, sense=GRB.GREATER_EQUAL, rhs=ub_normal_length[2],
#                          name='C_J')



#################
### OBJECTIVE ###
#################

# Defining objective function     
obj = LinExpr() 

for i in range(C+S+D):
     for j in range(C+S+D):
          obj = obj + c[i,j]*x[i,j] # x[i,j] loopt van x[0,1] tot x[4,5]

for i in range(C):
     for j in range(S):
          obj = obj + P*y[i,j]

for i in range(C+S):
     obj = obj + G*x[0,i]

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

# Here the model is actually being optimized
# model.optimize()

# Saving our solution in the form [name of variable, value of variable]
# solution = []
# for v in model.getVars():
#      solution.append([v.varName,v.x])
     
# print(solution)