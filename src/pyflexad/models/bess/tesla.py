from pyflexad.physical.stationary_battery import BESSHardware

power_wall_2 = BESSHardware(name="Tesla Powerwall 2",
                            max_capacity=13.5,
                            min_capacity=0.0,
                            max_charging_power=5.0,
                            max_discharging_power=-5.0,
                            self_discharge_factor=1.0)

power_wall_3 = BESSHardware(name="Tesla Powerwall 3",
                            max_capacity=13.5,
                            min_capacity=0.0,
                            max_charging_power=11.5,
                            max_discharging_power=-11.5,
                            self_discharge_factor=1.0)

power_wall_plus = BESSHardware(name="Tesla Powerwall+",
                               max_capacity=13.5,
                               min_capacity=0.0,
                               max_charging_power=5.8,
                               max_discharging_power=-5.8,
                               self_discharge_factor=1.0)

models = [power_wall_2, power_wall_3, power_wall_plus]
