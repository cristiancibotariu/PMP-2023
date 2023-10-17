from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import networkx as nx

# Defining the model structure. We can define the network by just passing a list of edges.
model = BayesianNetwork([('I', 'A'), ('C', 'A')])

# Defining individual CPDs.
cpd_i = TabularCPD(variable='I', variable_card=2, values=[[0.01], [0.99]]) # I = 0, !incendiu, I=1, incendiu
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.9995], [0.0005]]) # C=0 fara cutremur, C=1 cutremur

# The CPD for C is defined using the conditional probabilities based on U and R
cpd_a = TabularCPD(variable='A', variable_card=2, 
                   values=[[0.99, 0.01, 0.98, 0.02], 
                           [0.05, 0.95, 0.02, 0.98]],
                  evidence=['C', 'I'],
                  evidence_card=[2, 2])

# Associating the CPDs with the network
model.add_cpds(cpd_i, cpd_c, cpd_a)

assert model.check_model()

infer = VariableElimination(model)
result = infer.query(variables=['C'], evidence={'A': 1})
print(result)

pos = nx.circular_layout(model)
nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()