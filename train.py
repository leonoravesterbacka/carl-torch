from ml import RatioEstimator
from ml import Loader

loading = Loader()

x, y = loading.loading(
    folder='./data/samples',
    filename='train',
)
estimator = RatioEstimator(
    n_hidden=(10,10),
    activation="sigmoid"
)
estimator.train(
    method='carl',
    batch_size = 10,
    x='data/samples/x_train.npy',
    y='data/samples/y_train.npy',
    alpha=10,
    n_epochs=1,
)

estimator.save('models/carl')
