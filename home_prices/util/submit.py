from home_prices.dataload import load_test_data
import pandas as pd


def create_submission(model, name):
    test_features = load_test_data()
    test_indices = test_features.index
    x = test_features.to_numpy()
    predictions = model.predict(x)
    df = pd.DataFrame(predictions, index=test_indices)
    df.to_csv('../data/submissions/{}.csv'.format(name), header=["SalePrice"], index=True, float_format='%.2f')
