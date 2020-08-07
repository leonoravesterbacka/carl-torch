import logging
import optparse
import skorch
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV
from torch import nn
import torch
import numpy as np
from ml import RatioEstimator
from ml import Loader
from ml.utils.tools import load

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples', action='store', type=str, dest='samples', default='sherpaVsMG5', help='samples to derive weights for. default Sherpa vs. Madgraph5')
(opts, args) = parser.parse_args()
do = opts.samples



class RatioModel(nn.Module):

    def __init__(self, n_observables=9, n_hidden=(10, 10), activation="sigmoid", dropout_prob=0.0):

        super(RatioModel, self).__init__()

        # Save input
        self.n_hidden = n_hidden
        self.activation = "sigmoid"
        self.dropout_prob = dropout_prob

        # Build network
        self.layers = nn.ModuleList()
        n_last = n_observables

        # Hidden layers
        for n_hidden_units in n_hidden:
            if self.dropout_prob > 1.0e-9:
                self.layers.append(nn.Dropout(self.dropout_prob))
            self.layers.append(nn.Linear(n_last, n_hidden_units))
            n_last = n_hidden_units

        # Log r layer
        if self.dropout_prob > 1.0e-9:
            self.layers.append(nn.Dropout(self.dropout_prob))
        self.layers.append(nn.Linear(n_last, 1))

    def forward(self, x):
        s_hat = x
        for i, layer in enumerate(self.layers):
            if i > 0:
                s_hat = torch.relu(s_hat)
                #s_hat = self.activation(s_hat)
            s_hat = layer(s_hat)
        s_hat = torch.sigmoid(s_hat)
        #r_hat = (1 - s_hat) / s_hat

        return s_hat


doWeights = True #if False, train model for hyperparameter search
loading = Loader()
x,y =loading.loading(
    folder='./data/',
    plot=True,
    do = do,
    randomize = True,
    save = False,
    correlation = False,
)
net = NeuralNetClassifier(
    RatioModel,
    criterion=nn.BCELoss,
    max_epochs=10,
    lr=0.1,
    iterator_train__shuffle=True,
   # n_observables = 10,
   # n_hidden = 2,
)
print("before x dtype",x.dtype)
x = x.astype(np.float32)
#y = y.astype(np.long)
print("y",y)
print("y dtype",y.dtype)
print("x dtype",x.dtype)
y = y.squeeze(1)
print("y",y)
print("x",x)
net.set_params(train_split=False, verbose=0)
params = {
    'lr': [0.0001, 0.001,0.01,0.1],
    'max_epochs': [10, 20,30, 40, 50],
    #'module__num_units': [10, 20],
}
gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy', verbose=2)
print("hej")
gs.fit(x, y)
print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
