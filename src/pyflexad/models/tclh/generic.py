from pyflexad.physical.therm_load_heating import TCLHHardware

domestic_hot_water_heater_1 = TCLHHardware(name="Domestic Hot Water Heater 1", C=0.6, R=0.8 * 1e3, p=3, cop=3)

models = [domestic_hot_water_heater_1]
