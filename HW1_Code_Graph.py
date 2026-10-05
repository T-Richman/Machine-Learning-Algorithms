from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
# import some data
# x1=sepal length, x2=sepal width, x3=petal length, x4=petal width
iris = datasets.load_iris()
# retrieve the data features
x = iris.data
# retrieve the data labels
y = iris.target 

# Gets the 70% training set from the data for each flower
# 70% of the data is 105, 105/3=35 per flower
def train_data(x,s):
    sts_tr=[]
    vcr_tr=[]
    vgn_tr=[]
    for i in range(0,train_size):
        sts_tr.append(x[i])
        vcr_tr.append(x[i+50])
        vgn_tr.append(x[i+100])
    sts_tr=np.array(sts_tr)
    vcr_tr=np.array(vcr_tr)
    vgn_tr=np.array(vgn_tr)
    fl_train=[sts_tr,vcr_tr,vgn_tr]
    return fl_train
# Gets the septal length and width of the given flower
def sep_data(flwr):
    sep_len=[]
    sep_wid=[]
    for i in range(0,len(flwr)):
        sep_len.append(flwr[i][0])
        sep_wid.append(flwr[i][1])
    sep_len=np.array(sep_len)
    sep_wid=np.array(sep_wid)
    sep=[sep_len,sep_wid]
    return sep
# Gets the petal length and width of the given flower
def ptl_data(flwr):
    ptl_len=[]
    ptl_wid=[]
    for i in range(0,len(flwr)):
        ptl_len.append(flwr[i][2])
        ptl_wid.append(flwr[i][3])
    ptl_len=np.array(ptl_len)
    ptl_wid=np.array(ptl_wid)
    ptl=[ptl_len,ptl_wid]
    return ptl
# Plots the septal data of all flowers
def sep_plot():
    plt.figure(1)
    plt.scatter(sts_sep[0],sts_sep[1],c='r')
    plt.scatter(vcr_sep[0],vcr_sep[1],c='b')
    plt.scatter(vgn_sep[0],vgn_sep[1],c='g')
    plt.title("Septal Plot")
    plt.xlabel("Septal Length")
    plt.ylabel("Septal Width")
# Plots the petal data of all flowers
def ptl_plot():
    plt.figure(2)
    plt.scatter(sts_ptl[0],sts_ptl[1],c='r')
    plt.scatter(vcr_ptl[0],vcr_ptl[1],c='b')
    plt.scatter(vgn_ptl[0],vgn_ptl[1],c='g')
    plt.title("Petal Plot")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")

train_size=35
flowers=train_data(x,train_size)
# Gets the septal data
sts_sep=sep_data(flowers[0])
vcr_sep=sep_data(flowers[1])
vgn_sep=sep_data(flowers[2])
sep_plot()

sts_ptl=ptl_data(flowers[0])
vcr_ptl=ptl_data(flowers[1])
vgn_ptl=ptl_data(flowers[2])
ptl_plot()


