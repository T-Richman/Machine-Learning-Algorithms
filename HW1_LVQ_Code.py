from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

def dist(a,b):
    kv_pair_ls = [];
    for i in range(0,len(a)):
        dist_ls = [0,0];
        val = 0;
        for j in range(0,4):
            val+=(b[j]-a[i][j])*(b[j]-a[i][j])
        val = float(np.sqrt(val))
        dist_ls[0]=val
        dist_ls[1]=i
        kv_pair_ls.append(dist_ls)
    return kv_pair_ls
def bubblesort(list_input):
    ls=list_input
    for i in range(0,len(ls)-1):
        for j in range(i+1,len(ls)):
            if(ls[j]<ls[i]):
                temp=ls[i]
                ls[i]=ls[j]
                ls[j]=temp
    return ls

train_size=35
test_size=15
features = []
labels = []
for a in range(0,len(x)):
    x_ind=np.random.randint(0,len(x)-1)
    features.append(x[x_ind])
    labels.append(y[x_ind])
    if(len(x)-1>1):
        x=np.delete(x,x_ind,axis=0)
        y=np.delete(y,x_ind,axis=0)
features.append(x[0])
labels.append(y[0])
x=np.delete(x,0,axis=0)
y=np.delete(y,0,axis=0)
features.append(x[0])
labels.append(y[0])
x=np.delete(x,0,axis=0)
y=np.delete(y,0,axis=0)

flowers_tr=[]
for i in range(0,train_size*3):
    flowers_tr.append(np.append(features[0],labels[0]))
    features.pop(0)
    labels.pop(0)
flowers_tr=np.array(flowers_tr)
flowers_ts=[]
for i in range(0,test_size*3):
    flowers_ts.append(np.append(features[0],labels[0]))
    features.pop(0)
    labels.pop(0)
flowers_ts=np.array(flowers_ts)
#==============================================================================
print("k is always 1 for LVQ")
cv_cnt=int(input("Enter the number of codebook vectors: "))
alpha=float(input("Enter the learning rate: "))
epoch=int(input("Enter the epochs to occur: "))
correct_cnt=0

#gets random training variable to start codebook vectors
cv=[]
used_tr=[]
for i in range(0,cv_cnt):
    cv_ind=np.random.randint(0,len(flowers_tr))
    cv.append(flowers_tr[cv_ind])

#training the codebook vectors with each training datapoint
#repeats for the number of epochs
for e in range(0,epoch):
    for a in range(0,len(flowers_tr)):
        lr=alpha*(1-(e/epoch))
        val=dist(cv, flowers_tr[a])
        ord_val=bubblesort(np.transpose(val)[0])
        ind=0;
        for j in range(0,len(val)):
            if(ord_val[0]==val[j][0]):
                ind=val[j][1]
                break
        cv_label=(cv[ind][4])
        if(cv_label==flowers_tr[a][4]):
            for t in range(0,4):
                cv[ind][t]=cv[ind][t]+(lr*(flowers_tr[a][t]-cv[ind][t]))
        else:
            for t in range(0,4):
                cv[ind][t]=cv[ind][t]-(lr*(flowers_tr[a][t]-cv[ind][t]))

for i in range(0,len(flowers_ts)):
    #calculates and sorts nearest distances for the specific test datapoint
    val=dist(cv, flowers_ts[i])
    ord_val=bubblesort(np.transpose(val)[0])
    ind=0
    label=[]
    #k value for the number of neighbors is always 1 for LVQ
    #searches for the matching distance in distance-index list
    #gets index to search for it in original features list
    for j in range(0,len(val)):
        if(ord_val[0]==val[j][0]):
            ind=val[j][1]
            break
    final_label=(cv[ind][4])
    
    match label:
        case 0:
            final_label=0
            print("Setosa")
        case 1:
            final_label=1
            print("Versicolor")
        case 2:
            final_label=2
            print("Virginica")
    
    if(final_label==flowers_ts[i][4]):
        correct_cnt+=1
        print("Correct")
    else:
        print("Incorrect")
    print("")
accuracy=(correct_cnt/len(flowers_ts))*100
print("Accuracy: ",accuracy)
