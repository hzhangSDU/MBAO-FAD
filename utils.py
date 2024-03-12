
import math
import numpy
from  matplotlib import pyplot as plt
import matplotlib
plt.style.use('seaborn')
plt.rc('font', family = 'Times New Roman')

def loss_plot(train_loss, test_loss, time_loss, dataset, algorithm):
    
    fig, ax1 = plt.subplots()
    
    x = numpy.linspace(1, len(train_loss)+1, len(train_loss))
        
    ax1.plot(x, train_loss, color = 'purple', alpha=1.0, linewidth=1, label="training loss")
    ax1.plot(x, test_loss, color = 'darkcyan', alpha=1.0, linewidth=1, label="testing loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("RMSE loss")
    ax1.grid(False)  # Disable the grid for ax1

    ax2 = ax1.twinx()
    ax2.plot(x, time_loss, color = "#FF8C00", alpha=1.0, linewidth=1, label="time cost")
    ax2.set_ylabel("time cost(second)")
    ax2.grid(False)  # Disable the grid for ax1
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    
    plt.savefig('./pic/' + dataset + '_' + algorithm + '.png',dpi = 600)
    plt.show()


# Convert from a scalar index of the rule to a vector index of MFs. 
# This code refers to https://github.com/drwuHUST/MBGD_RDA.
def IDX2VEC(idx, nMFsVec, nMFs, M):

    vec = []
    for i in range(numpy.shape(nMFsVec)[0]):
        vec.append(0)

    prods = numpy.ones(shape=(M+1, 1))
    for times in range(M+1):
        prods[times] = 2 ** times

    if idx > prods[M]:
        print("Error: idx is larger than the number of rules.")

    prev = 0
    for MFs in range(numpy.shape(nMFsVec)[0]):
        vec[MFs] = math.floor((idx - 1 - prev) / prods[M - 1 - MFs]) + 1
        prev = prev + (vec[MFs] - 1) * prods[M - 1 - MFs]

    return vec

