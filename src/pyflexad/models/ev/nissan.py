from pyflexad.physical.electric_vehicle import EVHardware

nissan_leaf_6_6_kW_ac = EVHardware(name="Nissan Leaf 6.6kW AC",
                                   max_capacity=39,
                                   min_capacity=0.0,
                                   max_charging_power=6.6,
                                   max_discharging_power=-6.6,
                                   self_discharge_factor=1.0)

models = [nissan_leaf_6_6_kW_ac]
