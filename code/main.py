# Loading packages that are used in the code
import numpy as np
import pandas as pd
from gurobipy import Model,GRB,LinExpr

###################
### MODEL SETUP ###
###################

model = Model()

#################
### VARIABLES ###
#################

# Variables are defined as dictionaries, using the same indices you would
# use in your MILP formulation
x = {}
x[1] = model.addVar(lb=0, ub=1, vtype=GRB.INTEGER, name='x1')
x[2] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x2')

# Update the model (important!) so all variables are added to the model  
model.update()

###################
### CONSTRAINTS ###
###################



#################
### OBJECTIVE ###
#################

# Defining objective function     
obj = LinExpr() 

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
model.optimize()

# Saving our solution in the form [name of variable, value of variable]
solution = []
for v in model.getVars():
     solution.append([v.varName,v.x])
     
print(solution)