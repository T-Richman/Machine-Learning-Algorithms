import pandas as pd
import numpy as np
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