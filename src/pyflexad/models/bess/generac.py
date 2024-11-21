from pyflexad.physical.stationary_battery import BESSHardware

pwrcell_m3 = BESSHardware(name="GENERAC PWRcell M3",
                          max_capacity=9.0,
                          min_capacity=0.0,
                          max_charging_power=3.4,
                          max_discharging_power=-3.4,
                          self_discharge_factor=1.0)

pwrcell_m4 = BESSHardware(name="GENERAC PWRcell M4",
                          max_capacity=12.0,
                          min_capacity=0.0,
                          max_charging_power=4.5,
                          max_discharging_power=-4.5,
                          self_discharge_factor=1.0)

pwrcell_m5 = BESSHardware(name="GENERAC PWRcell M5",
                          max_capacity=15.0,
                          min_capacity=0.0,
                          max_charging_power=5.6,
                          max_discharging_power=-5.6,
                          self_discharge_factor=1.0)

pwrcell_m6 = BESSHardware(name="GENERAC PWRcell M6",
                          max_capacity=18.0,
                          min_capacity=0.0,
                          max_charging_power=6.7,
                          max_discharging_power=-6.7,
                          self_discharge_factor=1.0)

models = [pwrcell_m3, pwrcell_m4, pwrcell_m5, pwrcell_m6]
