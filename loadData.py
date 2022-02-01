from oscillatorDB import mongoMethods as mm
import tellurium as te
import pandas as pd

query = {"num_nodes": 3, "manually_checked":True, "oscillator": False, "simulates":True}
result, length = mm.query_database(query, returnLength=True)


# Manually check models
for i in range(688,length):
    print(i)
    model = result[i]["model"]
    ID = result[i]["ID"]
    m = te.loada(model)

    try:
        j = m.getFullJacobian()
        j = j.flatten()
        data = {"jacobian": [j],
                "oscillator": [False],
                "ID": ['a'+str(ID)]} # hacky bullshit to get it to save as a string
        oscillators = pd.DataFrame(data=data)
        oscillators.to_csv(path_or_buf="/home/hellsbells/Desktop/ID-2non-oscillator_jacobians2.csv", mode='a', header=False)
    except Exception as e:
        print(e)
        continue

print("done")


