from oscillatorDB import mongoMethods as mm
import tellurium as te
import matplotlib
# matplotlib.use('Qt5Agg')

ID = '7976231060579558765'

r = mm.query_database({"ID":ID})
model = r[0]

ant = model["model"]
m = te.loada(ant)
r = m.simulate()
m.plot()