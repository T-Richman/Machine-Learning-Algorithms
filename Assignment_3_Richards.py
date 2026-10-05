from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticReg():
    def __init__(self):
        print("Logistic Regression algorithm created")
        
    alpha=float(input("Enter the learning rate: "))
    epoch=int(input("Enter the epoch amount: "))

    data = load_breast_cancer() 

    X_raw=data.data
    Y_raw=data.target

    X=(np.subtract(X_raw,X_raw.min(axis=0)))/(np.subtract(X_raw.max(axis=0),X_raw.min(axis=0)))
    Y=(np.subtract(Y_raw,Y_raw.min(axis=0)))/(np.subtract(Y_raw.max(axis=0),Y_raw.min(axis=0)))

    Y_label=list(data.target_names)
    #weights "b" with last weight as bias "a"
    b=np.zeros(len(X[0])+1)
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    for i in range(0,int(len(X)*0.7)):
        ls=list(X[i])
        ls.append(1)
        X_train.append(np.array(ls))
        Y_train.append(float(Y[i]))
    for i in range(int(len(X)*0.7),len(X)):
        ls=list(X[i])
        ls.append(1)
        X_test.append(np.array(ls))
        Y_test.append(float(Y[i]))
    X_train=np.array(X_train)
    Y_train=np.array(Y_train)
    X_test=np.array(X_test)
    Y_test=np.array(Y_test)

    for e in range(0,epoch):
        #logits of the inputs
        l=X_train@b
        #odds of the inputs
        o=np.e**l
        #the signmoid function
        p=o/(1+o)
        #cost/error/loss of comparing expected (y) and actual (y_hat,p) probability
        loss=-Y_train*np.log(p) - (1-Y_train)*np.log(1-p)
        avg_loss=float((1/len(X_train))*sum(loss))
        
        grad_desc=0
        for i in range(0,len(X_train)):
            grad_desc+=((p[i]-Y_train[i])*X_train[i])
        grad_desc=grad_desc/len(X_train)
        b=b-alpha*grad_desc
        #print("Average Loss:",avg_loss)

    val=100
    acc=0
    for i in range(0, len(X_test)):
        print("X_test datapoint: ",i)
        l=X_test[i]@b
        o=np.e**l
        p=o/(1+o)
        if(p<0.5):
            print("Malignant")
            val=0
        elif(p>=0.5):
            print("Benign")
            val=1
        
        if(val==Y_test[i]):
            acc+=1
    acc=(acc/len(X_test))*100
    print("Accuracy: ",acc,"%")

class TimeSeries():
    def __init__(self):
        print("Time Series algorithm created")
    
    def show_data(price, days):
        #finding trend w/ linear regression
        x_mean=np.mean(days)
        y_mean=np.mean(price)
        sum_1=np.dot(price,days)
        sum_2=np.sum(days**2)
        sum_3=np.sum(days)
        a=float((sum_1-(y_mean*sum_3))/(sum_2-(x_mean*sum_3)))
        b=float(((y_mean*sum_2)-(x_mean*sum_1))/(sum_2-(x_mean*sum_3)))
        y_pred=a*days+b
        plt.figure()
        plt.scatter(days,price,c='black')
        plt.plot(days,price,c='blue')
        plt.plot(days,y_pred,c='red')
    def show_corr(acf, pacf, k_val, cb):
        plt.figure()
        plt.axhline(0,c='grey')
        plt.axhline(cb,c='red')
        plt.axhline(-cb,c='red')
        plt.scatter(k_val,acf)
        plt.title("ACF Values and Confidence Bands")
        plt.figure()
        plt.axhline(0,c='grey')
        plt.axhline(cb,c='red')
        plt.axhline(-cb,c='red')
        plt.scatter(k_val,pacf)
        plt.title("PACF Values and Confidence Bands")

    def differencing(x,y,start,end):
        diff_price=[]#[float(price[0]-price[0])]
        x_val=[]
        for i in range(start,end):
            diff_price.append(float(y[i]-y[i-1]))
            x_val.append(x[i])
        ls=[diff_price,x_val]
        plt.figure()
        plt.scatter(x_val,diff_price,c='black')
        plt.plot(x_val,diff_price,c='green')
        return ls
    def ACF_func(diff_price,max_lag):
        diff_y_mean=np.mean(diff_price)
        acf=[]
        for k in range(1,max_lag+1):
            num=0
            den=0
            for t in range(k,len(diff_price)):
                num+=((diff_price[t]-diff_y_mean)*(diff_price[t-k]-diff_y_mean))
            for t in range(0,len(diff_price)):
                den+=((diff_price[t]-diff_y_mean)**2)
            acf.append(float(num/den))
        acf=np.array(acf)
        return acf
    def PACF_Func(acf,max_lag):
        k_val=[]
        phi=[]
        for k in range(1,max_lag+1):    
            phi_j=[]
            sum_num=0
            sum_den=0
            if(k==1):
                phi_j.append(acf[k-1])
            else:
                for j in range(1,k):
                    #use previous lag's coefficients, j-1 for python list
                    sum_num+=phi[k-2][j-1]*acf[k-j-1]
                    sum_den+=phi[k-2][j-1]*acf[j-1]
                phi_kk=(float((acf[k-1]-sum_num)/(1-sum_den)))
                for j in range(1,k):
                    phi_j.append(phi[k-2][j-1]-phi_kk*phi[k-2][k-j-1])
                phi_j.append(phi_kk)
            phi.append(phi_j)
            k_val.append(k)
        pacf=[]
        for i in range(1, max_lag+1):
            pacf.append(phi[i-1][i-1])
        ls=[k_val,phi,pacf]
        return ls
    def pq_find(diff_price,acf,pacf,k_val):
        cb=(1.96/np.sqrt(len(diff_price)))
        p=0
        q=0
        for i in range(0,len(k_val)):
            if(acf[i]>cb or acf[i]<-cb):
                q+=1
            else:
                break
        for i in range(0,len(k_val)):
            if(pacf[i]>cb or pacf[i]<-cb):
                p+=1
            else:
                break
        ls=[p,q,cb]
        return ls
    def AR_func(diff_price,p):
        X=[]
        Y=[]
        #forming design matrix "X" for AR
        for t in range(p,len(diff_price)):
            temp=[1]
            for k in range(t-1,t-p-1,-1):
                temp.append(diff_price[k])
            X.append(temp)
            Y.append(diff_price[t])
        X=np.array(X)
        Y=np.array(Y)
        #finding coefficients for AR
        beta=(np.linalg.inv(X.T@X))@X.T@Y
        #residual for AR
        y_hat=X@beta
        err=Y-y_hat
        ls=[Y,y_hat,err]
        return ls
    def MA_func(q,Y,y_hat,err):
        err_mean=np.mean(err)
        theta=[]
        if(q==0):
            theta.append(0)
        else:
            for k in range(1,q+1):
                num=0
                den=0
                for t in range(k,len(err)):
                    num+=((err[t]-err_mean)*(err[t-k]-err_mean))
                for t in range(0,len(err)):
                    den+=((err[t]-err_mean)**2)
                theta.append(float(-num/den))
        theta=np.array(theta)
        epsilon=[]
        for i in range(0,len(err)):
            if(i==0):
                epsilon.append(err[0])
            else:
                epsilon.append(err[i]-err_mean-(theta[0]*epsilon[i-1]))

        ma=[]
        arima=[]
        for i in range(0,len(err)):    
            if(i==0):
                ma.append(epsilon[i])
            else:
                ma.append((theta[0]*epsilon[i-1]))
            arima.append(float(y_hat[i]+ma[i]))
        arima=np.array(arima)
        arima_err=Y-arima
        ls=[ma,arima,arima_err]
        return ls
    
    raw_df = pd.read_csv('40-Day_Gas_Price_Dataset.csv')
    data = np.hstack([raw_df.values[::, :]])
    date=[]
    price=[]
    days=[]
    for i in range(0,int(len(data)*0.7)):
        date.append(data[i][0])
        price.append(data[i][1])
        days.append(i+1)
    date=np.array(date)
    price=np.array(price)
    days=np.array(days)

    test_date=[]
    test_price=[]
    test_days=[]
    for i in range(int(len(data)*0.7),len(data)):
        test_date.append(data[i][0])
        test_price.append(data[i][1])
        test_days.append(i+1)
    test_date=np.array(test_date)
    test_price=np.array(test_price)
    test_days=np.array(test_days)

    #TRAINING
    #showing data
    show_data(price, days)
    #differencing
    ls_1=differencing(days,price,1,int(len(data)*0.7))
    diff_price=ls_1[0]
    x_val=ls_1[1]
    #ACF for q-value
    max_lag=int(len(diff_price)/3)
    acf=ACF_func(diff_price,max_lag)
    #PACF for p-value
    ls_2=PACF_Func(acf,max_lag)
    k_val=ls_2[0]
    phi=ls_2[1]
    pacf=ls_2[2]
    #finding p and q
    ls_3=pq_find(diff_price,acf,pacf,k_val)
    p=ls_3[0]
    q=ls_3[1]
    cb=ls_3[2]
    print("For the training set, p=",p," and q=",q)
    show_corr(acf, pacf, k_val, cb)
    #AR model for past values effect
    ls_4=AR_func(diff_price,p)
    Y=ls_4[0]
    y_hat=ls_4[1]
    err=ls_4[2]
    print("Actual Prices: ",Y)
    print("Predicted Prices: ",y_hat)
    print("Error: ",err)
    #MA model for past errors effect
    ls_5=MA_func(q,Y,y_hat,err)
    ma=ls_5[0]
    arima=ls_5[1]
    arima_err=ls_5[2]
    print("ARIMA Prediction: ",arima)
    print("Predicted Error: ",arima_err)
    
    #TESTING
    #showing data
    show_data(test_price, test_days)
    #differencing
    ls_1=differencing(test_days,test_price,1,int(len(data)*0.3))
    diff_price=ls_1[0]
    x_val=ls_1[1]
    #ACF for q-value
    max_lag=int(len(diff_price))
    acf=ACF_func(diff_price,max_lag)
    #PACF for p-value
    ls_2=PACF_Func(acf,max_lag)
    k_val=ls_2[0]
    phi=ls_2[1]
    pacf=ls_2[2]
    #finding p and q
    ls_3=pq_find(diff_price,acf,pacf,k_val)
    p=ls_3[0]
    q=ls_3[1]
    cb=ls_3[2]
    print("For the test set, p=",p," and q=",q)
    show_corr(acf, pacf, k_val, cb)
    #AR model for past values effect
    ls_4=AR_func(diff_price,p)
    Y=ls_4[0]
    y_hat=ls_4[1]
    err=ls_4[2]
    print("Actual Prices: ",Y)
    print("Prdicted Prices: ",y_hat)
    print("Error: ",err)
    #MA model for past errors effect
    ls_5=MA_func(q,Y,y_hat,err)
    ma=ls_5[0]
    arima=ls_5[1]
    arima_err=ls_5[2]
    print("ARIMA Prediction: ",arima)
    print("Predicted Error: ",arima_err)

LogisticReg()
TimeSeries()