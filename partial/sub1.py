from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import random
# import numpy as np
# from scipy import stats
# import arviz as az

runde = 0
vP0 = 0
vP1 = 0
token = 0

if random.random() >= 0.5:
    token = 1

while runde < 20000:
    m = 0
    n = 0
    if token == 1:
        if random.random() <= 0.5:
            n = 1
        if n == 1:
            for i in range(2):
                if random.random() <= 0.33:
                    m += 1
        else:
            if random.random() <= 0.33:
                    m += 1
    else:
        if random.random() <= 0.33:
            n = 1
        if n == 1:
            for i in range(2):
                if random.random() <= 0.5:
                    m += 1
        else: 
            if random.random() <= 0.5:
                    m += 1
    if n>m:
        if token == 0:
            vP0 += 1
        else: vP1 += 1
    else:
        if token == 0:
            vP1 += 1
        else: vP0 += 1
    runde += 1

if vP0 > vP1:
    print("Jucatorul P0 are mai multe sanse de victorie")
else: print("Jucatorul P1 are mai multe sanse de victorie")

model = BayesianNetwork([('coin_toss', 'P1'),('coint_toss','P0')])

coin_toss = TabularCPD(variable='coin_toss', variable_card=2,
                    values=[[0.5],[0.5]])

P0 = TabularCPD(variable='P0', variable_card=2,
                    values=[[0.33, 0.11, 0.22, 0.44], [0.66, 0.99, 0.88, 0.66]], #ori dai stema/ban, ori ss, ori sb, ori bb ori bs
                    evidence=['coin_toss'], evidence_card=[2])

P1 = TabularCPD(variable='P1', variable_card=2,
                    values=[[0.5, 0.25], [0.5, 0.75]],#aici din cauza ca nu e masluita, sansele de a se da ban stema sunt la fel cu cele de a da ban ban
                    evidence=['coin_toss'], evidence_card=[2])


model.add_cpds(coin_toss, P0, P1)

assert model.check_model()

infer = VariableElimination(model)
prob_P0 = infer.query(variables=['P0'], evidence={'P1' : 1})
print(prob_P0)

# prob_cutremur_stiind_alarma = infer.query(variables=['cutremur'], evidence={'alarma': 1})
# print(prob_cutremur_stiind_alarma)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()