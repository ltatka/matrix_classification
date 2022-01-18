from oscillatorDB import mongoMethods as mm
import tellurium as te
import pandas as pd

query = {"num_nodes": 3, "oscillator": False, "simulates": True}
result, length = mm.query_database(query, returnLength=True)



# for i in range(length):
#     r = te.loada(result[i]["model"])
#     try:
#         r.simulate()
#         newVal = {"simulates": True}
#         mm.update_model({"ID": result[i]["ID"]}, newVal)
#     except Exception:
#         newVal = {"simulates": False}
#         mm.update_model({"ID": result[i]["ID"]}, newVal)



#
# # ## Load and write out oscillators
# jacobians = []
# model_ids = []
# oscillator = []

for i in range(835, length):
    model = result[i]["model"]
    m = te.loada(model)
    try:
        j = m.getFullJacobian()
        j = j.flatten()
        data = {"jacobian": [j],
                "oscillator": [False]}
        oscillators = pd.DataFrame(data=data)
        oscillators.to_csv(path_or_buf="/home/hellsbells/Desktop/non-oscillator_jacobians.csv", mode='a', header=False)
    except Exception:
        continue

print("done")
#
# data = {"oscillator": oscillator,
#         "ID": model_ids,
#         "jacobian": jacobians}
#
# oscillators = pd.DataFrame(data=data)
#
# oscillators.to_csv(path_or_buf="/home/hellsbells/Desktop/non-oscillator_jacobians.csv")

