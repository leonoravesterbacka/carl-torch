from ml import RatioEstimator

carl = RatioEstimator()
carl.load('models/carl')

r_hat = carl.evaluate(x='data/samples/x_train.npy')
print("r_hat", r_hat)
