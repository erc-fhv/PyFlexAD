import pandas as pd
import numpy as np

from pyflexad.models.bess import bess_models
from pyflexad.models.ev import ev_models

from pyflexad.models.tclc import tclc_models
from pyflexad.models.tclh import tclh_models
from pyflexad.physical.pumped_hydro_energy_storage import PHESHardware


def main() -> None:
    bess_df = pd.DataFrame(bess_models)
    ev_df = pd.DataFrame(ev_models)
    tclc_df = pd.DataFrame(tclc_models)
    tclh_df = pd.DataFrame(tclh_models)

    phes_models = [PHESHardware(name="none", max_discharging_power=np.nan, max_charging_power=np.nan,
                                max_volume=np.nan, min_volume=np.nan, delta_h=np.nan), ]
    phes_df = pd.DataFrame(phes_models)

    print(bess_df.to_string(index=False))
    print()
    print(ev_df.to_string(index=False))
    print()
    print(tclc_df.to_string(index=False))
    print()
    print(tclh_df.to_string(index=False))
    print()
    print(phes_df.to_string(index=False))


if __name__ == '__main__':
    main()
