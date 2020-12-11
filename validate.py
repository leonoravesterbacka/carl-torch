import os
import sys
import logging
import optparse
from ml import RatioEstimator
from ml.utils.loading import Loader
from ml.utils.tools   import load

parser = optparse.OptionParser(usage="usage: %prog [opts]", version="%prog 1.0")
parser.add_option('-s', '--samples',   action='store', type=str, dest='samples',   default='dilepton', help='samples to derive weights for. Sherpa 2.2.8 ttbar dilepton')
parser.add_option('-v', '--variation', action='store', type=str, dest='variation', default='QSFUP', help='variation to derive weights for. default QSF down to QSF up')
parser.add_option('-n', '--nentries',  action='store', type=str, dest='nentries',  default=1000, help='specify the number of events do do the training on, default None means full sample')
parser.add_option('-p', '--datapath',  action='store', type=str, dest='datapath',  default='/eos/atlas/unpledged/group-tokyo/users/tatsuya/TruthAOD/Temp/Tuples/', help='path to where the data is stored')

(opts, args) = parser.parse_args()
sample = opts.samples
var = opts.variation
n = opts.nentries
p = opts.datapath
logger = logging.getLogger(__name__)
logger.info(" Doing validation of weights trained with datasets: %s , generator variation: %s  with %s  events.", sample, var, n)
    
#carl-torch inference###
#get the weight from carl-torch (weightCT) evaluated on the same model used for carlAthena and the root file from carlAthena
eventVarsCT = ['Njets','MET']
eventVarsCA = ['Njets','MET','weight']
jetVars = ['Jet_Pt','Jet_Mass'] 
lepVars = ['Lepton_Pt']
xCT ,_ = load(f= p+'/test.root',events=eventVarsCT,jets=jetVars,leps=lepVars,n=int(n),t='Tree',do=sample)
xCT = xCT[sorted(xCT.columns)]
carl = RatioEstimator()
carl.load('models/'+sample+'/'+var+'_carl_2000001')
r_hat, s_hat = carl.evaluate(x=xCT.to_numpy())
weightCT = 1./r_hat

###carlAthena inference###
#load sample with weight infered from carlAthena
xCA, _ = load(f=p+'/test.root',events=eventVarsCA,jets=jetVars,leps=lepVars,n=int(n),t='Tree')
weightCA = xCA.weight

###compare weights###
# draw histograms comparing weight from carl-torch (weightCT) from weight infered through carlAthena (ca.weight)
loading = Loader()
loading.validate_result(weightCT=weightCT, 
                        weightCA=weightCA, 
                        do=sample,
                        var=var,
                        plot=True,
                        n=n,
                        path=p,
)
