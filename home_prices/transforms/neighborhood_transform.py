import numpy as np
import pandas as pd
import re

neighborhoods = """
Blmngtn	Bloomington Heights
Blueste	Bluestem
BrDale	Briardale
BrkSide	Brookside
ClearCr	Clear Creek
CollgCr	College Creek
Crawfor	Crawford
Edwards	Edwards
Gilbert	Gilbert
IDOTRR	Iowa DOT and Rail Road
MeadowV	Meadow Village
Mitchel	Mitchell
Names	North Ames
NoRidge	Northridge
NPkVill	Northpark Villa
NridgHt	Northridge Heights
NWAmes	Northwest Ames
OldTown	Old Town
SWISU	South & West of Iowa State University
Sawyer	Sawyer
SawyerW	Sawyer West
Somerst	Somerset
StoneBr	Stone Brook
Timber	Timberland
Veenker	Veenker
"""


class NeighborhoodTransform:
    def __init__(self):
        parsed_neighborhoods = []
        for line in neighborhoods.split("\n"):
            if line == "":
                continue
            parts = re.split("\\s+", line)
            # parts = line.split(" ")
            n = parts[0].lower()
            parsed_neighborhoods.append(n)
        self.neighborhoods = np.array(parsed_neighborhoods)

    def transform_neighborhood_to_ordinal(self, neighborhood: str) -> float:
        if isinstance(neighborhood, float):
            return neighborhood
        elif isinstance(neighborhood, int):
            return float(neighborhood)
        elif not isinstance(neighborhood, str):
            return 0.0
        neighborhood = neighborhood.lower()
        j = self.neighborhoods.tolist().index(neighborhood)
        return float(j)

    def transform_neighborhood_array_to_ordinal(self, x):
        if isinstance(x, pd.DataFrame):
            t = x.applymap(self.transform_neighborhood_to_ordinal)
            return t
        elif isinstance(x, pd.Series):
            t = x.apply(self.transform_neighborhood_to_ordinal)
            return pd.DataFrame(t)
        elif isinstance(x, np.ndarray):
            t = np.array([[self.transform_neighborhood_to_ordinal(z[0])] for z in x])
            return t

    def transform(self, x, y=None):
        return self.transform_neighborhood_array_to_ordinal(x)

    def fit(self, x, y=None):
        return self.transform(x)

    def fit_transform(self, x, y=None):
        self.fit(x)
        return self.transform(x)
