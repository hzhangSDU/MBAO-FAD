


class config(object):
    
    def __init__(self, dataset_name):
        
        # dataset
        self.dataset_name = dataset_name
        self.dataset = eval(dataset_name)()
        self.path = self.dataset.path  # the path of this dataset.
        self.v = 0.7  # ratio of train set to test set
        
        # TSK-FS model
        self.M = self.dataset.M  # the input dimension of this dataset.
        self.nMFs = 2  # the number of Gaussian MFs in each input domain.
        
        # MBFAD-EAO algorithm
        self.P = 0.7  # the DropRule rate in MBFAD-EAO algorithm.
        self.batch_size = 64  # batch size of MBFAD-EAO algorithm.
        self.epochs = 500  # number of iterations of MBFAD-EAO algorithm.
        self.lr = 1e-3  # initial learning rate of MBFAD-EAO algorithm.
        
        self.beta1 = 0.9  # beta parameter of MBFAD-EAO algorithm.
        self.beta2 = 0.999  # beta parameter of MBFAD-EAO algorithm.
        self.beta3 = 0.99999  # beta parameter of MBFAD-EAO algorithm.
        self.eta1 = 1e-30  # eta parameter of MBFAD-EAO algorithm.
        self.eta2 = 1e-16  # eta parameter of MBFAD-EAO algorithm.
        self.gamma = 0.9  # gamma parameter of MBFAD-EAO algorithm.
        self.d = 2  # d parameter of MBFAD-EAO algorithm.


class PM10(object):
    
    def __init__(self):
        self.path = './dataset/PM10.mat'  # the path of this dataset.
        self.M = 7  # the input dimension of this dataset.

