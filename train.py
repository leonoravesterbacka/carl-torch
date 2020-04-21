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
    batch_size = 10,
    x='data/samples/x_train.npy',
    y='data/samples/y_train.npy',
    alpha=1.0,
    n_epochs=1,
)

estimator.save('models/carl')
