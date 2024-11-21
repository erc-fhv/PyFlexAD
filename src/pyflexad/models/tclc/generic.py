from pyflexad.physical.therm_load_cooling import TCLCHardware

air_conditioner_1 = TCLCHardware(name="Air Conditioner 1", C=2.0, R=2, p_max=5, cop=2.5)

models = [air_conditioner_1]
