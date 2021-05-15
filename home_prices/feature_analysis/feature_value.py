

class FeatureAndValue:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

    def add(self, x: float):
        self.value += x

    def __str__(self):
        return "{} {}".format(self.name, self.value)
