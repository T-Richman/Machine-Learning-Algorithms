from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class knn:
    def __init__(self):
        print("knn algorithm created")
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

class lvq:
    def __init__(self):
        print("lvq algorithm created")
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

class linreg:
    def __init__(self):
        print("linear regression algorithm created")
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]]) 
    target = raw_df.values[1::2, 2] 

    data_norm=(np.subtract(data,data.min(axis=0)))/(np.subtract(data.max(axis=0),data.min(axis=0)))
    target_norm=(np.subtract(target,target.min(axis=0)))/(np.subtract(target.max(axis=0),target.min(axis=0)))

    alpha=float(input("Enter the learning rate: "))
    epoch=int(input("Enter the epoch amount: "))

    #separating training and testing data
    #also adding bias as ones column
    train_size=int(len(data_norm)*0.7)
    test_size=len(data_norm)-train_size
    x=[]
    y=[]
    for i in range(0,train_size):
        x.append(np.insert(data_norm[i],0,1))
        y.append(target_norm[i])
    x=np.array(x)
    y=np.array(y)

    #exact values of w, but doesnt work if x_t*x is singular
    w=np.random.rand(14)*0.01
    for e in range(0,epoch):
        y_hat=x @ w
        res=np.subtract(y_hat,y)
        mse=float((1/2)*np.mean(res**2))
        #gradient descent formula, is the slope for higher dimensions
        grad_desc=(np.transpose(x) @ res)/len(y)
        w=np.subtract(w,grad_desc*alpha)
        print(mse)
    print("Complete")
    print("Mean Squared Error (Training Dataset):",mse)

    x_ts=[]
    y_ts=[]
    for i in range(train_size,train_size+test_size):
        x_ts.append(np.insert(data_norm[i],0,1))
        y_ts.append(target_norm[i])
    x_ts=np.array(x_ts)
    y_ts=np.array(y_ts)

    y_ts_hat=x_ts @ w
    res_ts=np.subtract(y_ts_hat,y_ts)
    mse_ts=float((1/2)*np.mean(res_ts**2))

    print("Mean Squared Error (Test Dataset):",mse_ts)
    
knn()
lvq()
linreg()