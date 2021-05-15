
def map_kitchen_quality(quality: str) -> float:
    if quality == "Ex":
        return 4.
    elif quality == "Gd":
        return 3.
    elif quality == "TA":
        return 2.
    elif quality == "Fa":
        return 1.
    return 0.


def map_functional(f: str) -> float:
    if f == 'Typ':
        return 10.
    elif f == 'Min1':
        return 9.
    elif f == 'Min2':
        return 8.
    elif f == 'Mod':
        return 5.5
    elif f == 'Maj1':
        return 3.
    elif f == 'Maj2':
        return 2.
    elif f == 'Sev':
        return 1.
    elif f == 'Sal':
        return 0.
    return 8.
