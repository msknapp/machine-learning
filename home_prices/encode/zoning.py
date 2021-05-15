import pandas as pd
import numpy as np


def zoning_to_ordinal(zone: str) -> float:
    # This converts a zone into a number, in a way that the distance between them makes more sense.
    if isinstance(zone, float):
        return zone
    elif isinstance(zone, int):
        return float(int)
    elif isinstance(zone, np.float32):
        return zone
    if zone == "A":
        return 0.
    elif zone == "RL":
        return 1.
    elif zone == "RP":
        return 1.2
    elif zone == "RM":
        return 2.
    elif zone == "RH":
        return 3.
    elif zone == "FV":
        return 5.
    elif zone.startswith("C"):
        return 8.
    elif zone == "I":
        return 10.
    return 4.


def zone_array_to_ordinal(zone):
    if isinstance(zone, pd.Series):
        t = zone.apply(zoning_to_ordinal)
        # even though the input zone is just one series, we must produce a data frame
        # the output must be two dimensional
        return pd.DataFrame(t)
    elif isinstance(zone, pd.DataFrame):
        t = zone.applymap(zoning_to_ordinal)
        return t
    elif isinstance(zone, np.ndarray):
        # the output must be a list of lists, otherwise it will be considered one dimensional and the
        # column transformer will throw an error
        t = np.array([[zoning_to_ordinal(z[0])] for z in zone])
        # t.reshape([len(t), 1])
        return t
