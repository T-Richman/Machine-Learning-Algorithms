from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target

#find euclidean distance between training(a) and test(b) datapoints
def dist(a,b):
    #Stores the training datapoint index and distance
    #as key-value pair
    kv_pair_ls = [];
    #goes through all of the training datapoints
    for i in range(0,len(a)):
        #Stores the distances and index of the respective training datapoint
        dist_ls = [0,0];
        val = 0;
        #uses each feature from the training and test datapoint for distance
        for j in range(0,len(a[i])-1):
            val+=(b[j]-a[i][j])*(b[j]-a[i][j])
        val = float(np.sqrt(val))
        dist_ls[0]=val
        dist_ls[1]=i
        #adds distance and index to list of all distance-index
        kv_pair_ls.append(dist_ls)
        #print(dist_ls)
        #print("")
    return kv_pair_ls

#bubble sort to get the closest distances at the front of the list
def bubblesort(list_input):
    ls=list_input
    for i in range(0,len(ls)-1):
        for j in range(i+1,len(ls)):
            if(ls[j]<ls[i]):
                temp=ls[i]
                ls[i]=ls[j]
                ls[j]=temp
    return ls

#Sets the training and test data amounts
train_size=35
test_size=15

#randomizes the dataset
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

# Separates the training and test data 
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


k_input=int(input("Enter k-neighbors: "))
# cv_cnt=int(input("Enter the number of codebook vectors: "))
# alpha=int(input("Enter the learning rate: "))
# epoch=int(input("Enter the epochs to occur: "))
correct_cnt=0


#performs test on each test datapoint
for i in range(0,len(flowers_ts)):
    #calculates and sorts nearest distances for the specific test datapoint
    val=dist(flowers_tr, flowers_ts[i])
    ord_val=bubblesort(np.transpose(val)[0])
    ind=0
    label=[]
    #k value for the number of neighbors to use/check
    for k in range(0,k_input):
        #searches for the matching distance in distance-index list
        #gets index to search for it in original features list
        for j in range(0,len(val)):
            if(ord_val[k]==val[j][0]):
                ind=val[j][1]
                break
        #print("Index: ",ind)
        label.append(flowers_tr[ind][4])
    #print("Flower Labels: ",label)
    
    sts_l=0
    vcr_l=0
    vgn_l=0
    final_label=0
    for l in range(0,len(label)):
        match label[l]:
            case 0:
                sts_l+=1
            case 1:
                vcr_l+=1
            case 2:
                vgn_l+=1
    
    if(sts_l>vcr_l and sts_l>vgn_l):
        final_label=0
        print("Setosa")
    elif(vcr_l>sts_l and vcr_l>vgn_l):
        final_label=1
        print("Versicolor")
    elif(vgn_l>sts_l and vgn_l>vcr_l):
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
