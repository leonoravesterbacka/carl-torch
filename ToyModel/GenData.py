import ROOT
import numpy as np

# Helper function to create an example tree
def make_data(nevts = 500000):
    #root_file = ROOT.TFile("Nominal.root", "RECREATE")
    root_file = ROOT.TFile("Alt.root", "RECREATE")
    tree = ROOT.TTree("ThreeDGauss", "ThreeDGauss")
    
    # Input features
    x = np.empty((1), dtype="float32")
    y = np.empty((1), dtype="float32")
    z = np.empty((1), dtype="float32")
    # Event Weights
    w = np.empty((1), dtype="float32")
    # Toggle of gaussian modes
    t0 = np.empty((1), dtype="float32")
    t1 = np.empty((1), dtype="float32")

    # Add features and weights to tree
    tree.Branch("x", x, "x/F")
    tree.Branch("y", y, "y/F")
    tree.Branch("z", z, "y/F")
    tree.Branch("w", w, "y/F")
    tree.Branch("t0", t0, "y/F")
    tree.Branch("t1", t1, "y/F")
 
    # Random generate features and weights
    for i in range(nevts):
 
        ## Negative weight version
        ##w[0] = (np.ceil( np.random.random()  - 0.1 ) * 4) - 2  # Alt
        #w[0] = 2.0    # Nom 
        #xmu, xsigma = 0.0, 1.5 if w[0] > 0 else  0.25
        #ymu, ysigma = 0.0, 1.5 if w[0] > 0 else  0.25
        #zmu, zsigma = 0.0, 1.5 if w[0] > 0 else  0.25
        #
        #x[0] = np.random.normal(xmu, xsigma, 1)
        #y[0] = np.random.normal(ymu, ysigma, 1)
        #z[0] = np.random.normal(zmu, zsigma, 1)


        ## Composite positive weight version - gaussian mixture model
        t0[0] = 0.0 # Nom
        #t0[0] = np.ceil( np.random.random()  - 0.3 ) # Alt
        t1[0] = np.ceil( np.random.random()  - 0.5 )
        #print(" -------")
        #print("t0:   {}".format(t0))
        #print("t1:   {}".format(t1))
        # Weight
        #w[0] = 1.0 # Nom
        #w[0] = 1.0 if t0[0] > 0 else 0.5 # Alt
        w[0] = ( ( np.random.random() ) * 2) # Alt + Nominal = uniform sampling with mean = 1.0
        
        
        # x-axis parameters
        xmu, xsigma = 0.0, 0.6  # 0.0, 0.5 Alt # Nom 0.0, 1.0
        xmu_dis = 0.9 # 1.0 Atl # Nom 0.0
        xmu_neg, xsigma_neg = 0.0, 0.5
        # y-axis parameters
        ymu, ysigma = 0.0, 0.6 # 0.0, 0.5 Alt # Nom 0.0, 1.0
        ymu_dis = 0.9 # 1.0 Atl # Nom 0.0
        ymu_neg, ysigma_neg = 0.0, 0.5
        # z-axis paramters
        zmu, zsigma = 0.0, 0.6 # 0.0, 0.5 Alt # Nom 0.0, 1.0
        zmu_dis = 0.9 # 1.0 Atl # Nom 0.0
        zmu_neg, zsigma_neg = 0.0, 0.5

        #x[0] = t0[0]*t1[0]*np.random.normal(xmu-xmu_dis, xsigma, 1) + t0[0]*(1-t1[0])*np.random.normal(xmu+xmu_dis, xsigma, 1) #+ (1-t0[0])*np.random.normal(xmu_neg, xsigma_neg, 1)
        x[0] = t1[0]*np.random.normal(xmu-xmu_dis, xsigma, 1) + (1-t1[0])*np.random.normal(xmu+xmu_dis, xsigma, 1) #+ (1-t0[0])*np.random.normal(xmu_neg, xsigma_neg, 1)
        y[0] = t1[0]*np.random.normal(ymu-ymu_dis, ysigma, 1) + (1-t1[0])*np.random.normal(ymu+ymu_dis, ysigma, 1) #+ (1-t0[0])*np.random.normal(ymu_neg, ysigma_neg, 1)
        z[0] = t1[0]*np.random.normal(zmu-zmu_dis, zsigma, 1) + (1-t1[0])*np.random.normal(zmu+zmu_dis, zsigma, 1) #+ (1-t0[0])*np.random.normal(zmu_neg, zsigma_neg, 1)



        tree.Fill()
    root_file.Write()
 
    return (root_file, x, y, z, w), tree

#def plot_data():
    

def main():
    _, tree = make_data()
    
if __name__ == "__main__":
    main()
