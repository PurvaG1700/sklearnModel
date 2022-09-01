
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing


def train_model():
    housing = fetch_california_housing()
    df = pd.DataFrame(housing["data"], columns=housing["feature_names"])
    keep_vars = ['HouseAge', 'AveBedrms', 'Population', 'MedInc']
    x = pd.DataFrame(housing["data"], columns=housing["feature_names"])[
        keep_vars]
    y = pd.DataFrame(housing["target"], columns=housing["target_names"])
    model = LinearRegression()
    model.fit(x, y)
    print(model.score(x, y))


    pkl_filename = 'Cali_model.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)

    model.predict

if __name__ == '__main__':
    train_model()