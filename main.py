import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from polygon import RESTClient
from model import LSTMModel
import os


def visualize_data(pred, actual):
    plt.plot(pred, color="red")
    plt.plot(actual, color="blue")

    plt.suptitle("Predictions in red, actual in blue starting from August 30th, 2024")

    plt.xlabel("Days")

    plt.show()


def query_data(ticker, f, t):
    client = RESTClient("not provided")

    data = []
    for day in client.list_aggs(
        ticker=ticker,
        multiplier=1,
        timespan="day",
        from_=f,
        to=t,
        limit=50000
    ):
        data.append(day)

    return data


def create_dataset(data, lookback):
    X, y = [], []
    for i in range(len(data)-lookback):
        # pytorch expects the input features to be a 3d tensor, where each matrix in the tensor
        # is lookback # of feature arrays, where each feature array contains feature_num of values
        # (we only have one feature here right now, the price, so it's arrays of one value within arrays
        # all packaged into a tensor. add more features (rsi, etc.) later.)
        feature = [[data[j].open] for j in range(i, i + lookback)]  # given a range of opening prices,
        target = [[data[j].open] for j in range(i + 1, i + lookback + 1)]  # predict the following prices
        X.append(feature)
        y.append(target)
    """
    norm_X, norm_y = normalize_data(X, y)

    # wrap each value into a feature array
    X, y = [], []
    i = 0
    for ls in norm_X:  # really want to compress this to one ginormous list comprehension but
        # I can't figure out how
        X.append([[x] for x in ls])
    for ls in norm_y:
        y.append([[x] for x in ls])
    """

    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


"""
# fit data to range 0,1
def normalize_data(X, y):
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(X)
    scaler_y.fit(y)
    X = scaler_X.transform(X)
    y = scaler_y.transform(y)
    return X, y
"""


def preprocessing():
    # raw data from the Polygon API
    train_data = query_data(ticker="MSFT", f="2022-12-27", t="2024-08-01")
    test_data = query_data(ticker="MSFT", f="2024-08-02", t="2024-12-23")

    # normalized datasets consisting of len5 windows of data
    datasets = create_dataset(train_data, lookback=20), create_dataset(test_data, lookback=20)
    return datasets[0][0], datasets[0][1], datasets[1][0], datasets[1][1]


def main():
    trainX, trainY, testX, testY = preprocessing()

    # for i in range(len(testX)):
    #     print(f"{[feature.item() for tensor in testX[i] for feature in tensor]} --> {[feature.item() for tensor in testY[i] for feature in tensor]}")

    model = LSTMModel()

    if os.stat("model.pt").st_size == 0:
        trained_model = LSTMModel.train_model(model=model,
                                              X_train=trainX,
                                              y_train=trainY,
                                              X_test=testX,
                                              y_test=testY,
                                              num_epochs=1000
                                              )
        torch.save(trained_model.state_dict(), "model.pt")  # save to a local directory named "model"
    else:
        model.load_state_dict(torch.load("model.pt", weights_only=True))
        trained_model = model

    # reference for strategies for forecasting more than one timestep into the future:
    # https://machinelearningmastery.com/multi-step-time-series-forecasting/

    # Do inference on new data
    with torch.no_grad():
        trained_model.eval()
        predictions = trained_model(testX)
        predictions = [seq[-1][0] for seq in predictions]

    print(predictions)

    actual = []
    for seq in testX[20:]:
        actual.append(seq[0][0])

    visualize_data(predictions, actual)


if __name__ == "__main__":
    main()
