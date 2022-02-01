from oscillatorDB import mongoMethods as mm
import tellurium as te
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


query = {"num_nodes": 3, "oscillator": False, "manually_checked": False}
result, length = mm.query_database(query, returnLength=True)





# Manually check models
for i in range(length):
    ID = result[i]["ID"]
    print(f"{i}: Model {ID}")
    model = result[i]["model"]
    m = te.loada(model)
    try:
        r = m.simulate(0,1000,1000)
        m.plot()
        oscillator = mm.yes_or_no("Is this an oscillator?")
        if oscillator == None:
            r= m.simulate(0,10000,10000)
            m.plot()
            oscillator = mm.yes_or_no("Is this an oscillator?")

        newEntry = {"oscillator": oscillator, "manually_checked": True}
        mm.update_model({"ID": ID}, newEntry)

    except Exception:
        newEntry = {"simulates": False,
                    "manually_checked": True}
        mm.update_model({"ID": ID}, newEntry)
        continue

print("done")

#
