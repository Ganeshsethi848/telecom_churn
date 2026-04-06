import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset,random_split
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

device='cuda'
torch.manual_seed(123)

class Telecom_data(Dataset):
    def __init__(self):
        super(Telecom_data,self).__init__()
        self.df=pd.read_csv('/home/lonewolf8/Desktop/telecom_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        self.df.drop(self.df.iloc[:,:5],axis=1,inplace=True)
    
        #print(self.df)
        for i in self.df.columns[1:-3]:
            uniq=[i for i in enumerate(self.df[i].unique())]
            print(uniq)
            def f(x):
                for i in uniq:
                    if x==i[1]:
                        return np.int64(i[0])
                    
            self.df[i]=self.df[i].apply(f)

        def g(x):
            if x=='Yes':
                return 1
            else:
                return 0
            
        self.df['Churn']=self.df['Churn'].apply(g)
        
        def h(x):
            if x==' ':
                return 0.0
            else:
                return float(x)
            
        
        self.df['TotalCharges']=self.df['TotalCharges'].apply(h)
        self.X=self.df.drop(['Churn'],axis=1).values
        self.Y=self.df['Churn'].values


    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        
        x=self.X[index].tolist()
        y=self.Y[index].tolist()
        
        return torch.tensor(x).float(),torch.tensor(y).long()


ob=Telecom_data()
#print(ob[90])
train_len=int(0.8*len(ob))
test_len=len(ob)-train_len

train_dataset,test_dataset=random_split(ob,[train_len,test_len])

train_dl=DataLoader(train_dataset,batch_size=32,shuffle=True,drop_last=True)
test_dl=DataLoader(test_dataset,batch_size=32,shuffle=True,drop_last=True)
'''
#model
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.model=nn.Sequential(nn.Linear(15,32),
                            nn.ReLU(),
                            nn.Linear(32,64),
                            nn.ReLU(),
                            nn.Linear(64,128),
                            nn.ReLU(),
                            nn.Linear(128,2))
        
    def forward(self,x):
            
        return self.model(x)
        
mlp_model=MLP().to(device=device)
#print(mlp_model)
loss=nn.CrossEntropyLoss()
opt=optim.AdamW(mlp_model.parameters())

#train
def train(batch,model,opt,loss):
    model.train()
    opt.zero_grad()
    x,y=batch
    x=x.to(device);y=y.to(device)
    pred=mlp_model(x)
    l=loss(pred,y)
    l.backward()
    opt.step()

    return l.item()

#test
@torch.no_grad()
def test(batch,model):
    model.eval()
    x,y=batch
    x=x.to(device);y=y.to(device)
    pred=model(x)
    prob=torch.softmax(pred,dim=1)
    ind=torch.argmax(prob,dim=1)
    correct=((ind==y).sum())/len(y)

    return correct.item()

#epochs
train_loss=[];test_acc=[]
for epoch in range(1000):
    print(f"epoch: {epoch+1}")
    train_epoch_loss=[];test_epoch_acc=[]
    for batch in train_dl:
        loss_val=train(batch,mlp_model,opt,loss)
        train_epoch_loss.append(loss_val)

    train_loss.append(np.mean(train_epoch_loss))

    for batch in test_dl:
        acc=test(batch,mlp_model)
        test_epoch_acc.append(acc)  

    test_acc.append(np.mean(acc))
    print(f"loss_value: {np.mean(train_loss)} acc: {np.mean(test_acc)}")
print(f"loss_value: {np.mean(train_loss)} acc: {np.mean(test_acc)}")

torch.save(mlp_model.state_dict(),'/home/lonewolf8/Desktop/telecom_churn/telecom_data_model.pth')

'''
