from ml import RatioEstimator
from ml import Loader

loading = Loader()

x, y = loading.loading(
    folder='./data/samples',
    filename='train',
    plot=True
)
estimator = RatioEstimator(
    n_hidden=(10,10),
    activation="sigmoid"
)
estimator.train(
    method='carl',
    batch_size = 128,
    x='data/samples/x_train.npy',
    y='data/samples/y_train.npy',
    alpha=0.0001,
    n_epochs=20,
)

estimator.save('models/carl')
