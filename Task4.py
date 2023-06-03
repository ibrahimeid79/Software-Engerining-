import pandas as pd
import numpy as np
import pyomo.environ as pyo

# Load shipment data
shipments = pd.read_csv('shipments.csv')

# Define optimization model
model = pyo.ConcreteModel()

# Define decision variables
model.x = pyo.Var(range(len(shipments)), within=pyo.Binary)

# Define objective function
model.obj = pyo.Objective(expr=pyo.summation(shipments['Cost'], model.x), sense=pyo.minimize)

# Define constraints
model.constraints = pyo.ConstraintList()
for i in range(len(shipments)):
    model.constraints.add(pyo.summation(model.x[j] for j in range(len(shipments)) if shipments.loc[i, 'Origin'] == shipments.loc[j, 'Origin']) == 1)
    model.constraints.add(pyo.summation(model.x[j] for j in range(len(shipments)) if shipments.loc[i, 'Destination'] == shipments.loc[j, 'Destination']) == 1)

# Solve optimization model

solver = pyo.SolverFactory('glpk')
result = solver.solve(model)

# Print optimal solution
if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    print('Optimal solution found')
    for i in range(len(shipments)):
        if pyo.value(model.x[i]) == 1:
            print(f'Shipment {i+1} from {shipments.loc[i, "Origin"]} to {shipments.loc[i, "Destination"]}')
else:
    print('No optimal solution found')