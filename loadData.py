from oscillatorDB import mongoMethods as mm
import tellurium as te
import pandas as pd

query = {"num_nodes": 3, "oscillator": True}
result = mm.query_database(query)


## Load and write out oscillators
jacobians = []
model_ids = []
oscillator = []
for i in range(10):
    model = result[i]["model"]
    m = te.loada(model)
    model_ids.append(result[i]["ID"])
    j = m.getFullJacobian()
    j = j.flatten()
    jacobians.append(j)
    oscillator.append(result[i]["oscillator"])

data = {"oscillator": oscillator,
        "ID": model_ids,
        "jacobian": jacobians}

oscillators = pd.DataFrame(data=data)

oscillators.to_csv("/home/hellsbells/Desktop/oscillator_jacobains.csv")

#
# ## Load and write out non-oscillators
# query = {"num_nodes": 3, "oscillator": False}
# result = mm.query_database(query)
#
# for i in result.count():
#     model = result[i]["model"]
#     m = te.loada(model)
#     model_ids.append(result[i]["ID"])
#     j = m.getFullJacobian()
#     jacobians.append(j)
#     oscillator.append(result[i]["oscillator"])
#
# data = {"oscillator": oscillator,
#         "ID": model_ids,
#         "jacobian": jacobians}
#
