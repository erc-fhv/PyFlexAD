from pyflexad.physical.electric_vehicle import EVHardware

model_y_11kw_ac = EVHardware(name="Tesla Model Y 11kW AC",
                             max_capacity=57.5,
                             min_capacity=0.0,
                             max_charging_power=11.0,
                             max_discharging_power=-11.0,
                             self_discharge_factor=1.0)

model_y_100kw_dc = EVHardware(name="Tesla Model Y 100kW DC",
                              max_capacity=57.5,
                              min_capacity=0.0,
                              max_charging_power=100.0,
                              max_discharging_power=-100.0,
                              self_discharge_factor=1.0)

model_3_rwd_11kw_ac = EVHardware(name="Tesla Model 3 RWD 11kW AC",
                                 max_capacity=57.5,
                                 min_capacity=0.0,
                                 max_charging_power=11.0,
                                 max_discharging_power=-11.0,
                                 self_discharge_factor=1.0)

model_3_rwd_100kw_dc = EVHardware(name="Tesla Model 3 RWD 100kW DC",
                                  max_capacity=57.5,
                                  min_capacity=0.0,
                                  max_charging_power=100.0,
                                  max_discharging_power=-100.0,
                                  self_discharge_factor=1.0)

model_s_p110d_16_5kw_ac = EVHardware(name="Tesla Model S P100D 16.5kW AC",
                                     max_capacity=95.0,
                                     min_capacity=0.0,
                                     max_charging_power=16.5,
                                     max_discharging_power=-16.5,
                                     self_discharge_factor=1.0)

models = [model_y_11kw_ac, model_y_100kw_dc, model_3_rwd_11kw_ac, model_3_rwd_100kw_dc, model_s_p110d_16_5kw_ac]
