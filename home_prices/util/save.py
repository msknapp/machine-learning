from joblib import dump


def save_model(model, name: str):
    dump(model, '../saved_models/{}.joblib'.format(name))
