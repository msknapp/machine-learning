import pandas as pd
import numpy as np


def house_style_ordinal(style: str) -> float:
    if isinstance(style, float):
        return style
    elif isinstance(style, int):
        return float(style)
    elif isinstance(style, np.float32):
        return style
    if style == "1Story":
        return 1.
    elif style == "1.5Unf":
        return 1.3
    elif style == "1.5Fin":
        return 1.5
    elif style == "2Story":
        return 2.0
    elif style == "2.5Unf":
        return 2.3
    elif style == "2.5Fin":
        return 2.5
    elif style == "SFoyer":
        return 1.9
    elif style == "SLvl":
        return 1.8
    return 1.25


def house_style_array_to_ordinal(style):
    if isinstance(style, pd.Series):
        t = style.apply(house_style_ordinal)
        return pd.DataFrame(t)
    elif isinstance(style, pd.DataFrame):
        t = style.applymap(house_style_ordinal)
        return t
    elif isinstance(style, np.ndarray):
        # the output must be a list of lists, otherwise it will be considered one dimensional and the
        # column transformer will throw an error
        t = np.array([[house_style_ordinal(z[0])] for z in style])
        # t.reshape([len(t), 1])
        return t
