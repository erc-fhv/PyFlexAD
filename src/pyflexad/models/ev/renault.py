from pyflexad.physical.electric_vehicle import EVHardware

zoe_ze40_r110_22_kw_ac = EVHardware(name="Renault Zoe ZE40 R110 22kW AC",
                                    max_capacity=41.0,
                                    min_capacity=0.0,
                                    max_charging_power=22.0,
                                    max_discharging_power=-22.0,
                                    self_discharge_factor=1.0)

zoe_ze40_r110_40_kw_dc = EVHardware(name="Renault Zoe ZE40 R110 40kW DC",
                                    max_capacity=41.0,
                                    min_capacity=0.0,
                                    max_charging_power=40.0,
                                    max_discharging_power=-40.0,
                                    self_discharge_factor=1.0)

zoe_ze50_r135_22_kw_ac = EVHardware(name="Renault Zoe ZE50 R135 22kW AC",
                                    max_capacity=52.0,
                                    min_capacity=0.0,
                                    max_charging_power=22.0,
                                    max_discharging_power=-22.0,
                                    self_discharge_factor=1.0)

zoe_ze50_r135_41_kw_dc = EVHardware(name="Renault Zoe ZE50 R135 41kW DC",
                                    max_capacity=52.0,
                                    min_capacity=0.0,
                                    max_charging_power=41.0,
                                    max_discharging_power=-41.0,
                                    self_discharge_factor=1.0)

models = [
    zoe_ze40_r110_22_kw_ac,
    zoe_ze40_r110_40_kw_dc,
    zoe_ze50_r135_22_kw_ac,
    zoe_ze50_r135_41_kw_dc]
