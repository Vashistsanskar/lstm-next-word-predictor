import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset , DataLoader
from config import *
from preparing import X,y,vocab_size
from model import LSTMModel
torch.manual_seed(seed)

def main():
    # Datset and DataLoader classes
    class CustomDataset(Dataset):

        def __init__(self,X,y):
            self.X = X
            self.y = y

        def __len__(self):
            return self.X.shape[0]
        
        def __getitem__(self,idx):
            return self.X[idx],self.y[idx]
        
    dataset = CustomDataset(X,y)

    dataLoader = DataLoader(dataset,batch_size = 32,shuffle = True)

    #create model object

    model = LSTMModel(vocab_size=vocab_size,dropout=dropout)


    #send model to gpu
    model = model.to(device)

    #set parameters
    epochs = 50
    learning_rate = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr = learning_rate)

    # training loop

    for epoch in range(epochs):
        total_loss = 0
        for batch_X,batch_y in dataLoader:
            # send data to gpu
            batch_X,batch_y  = batch_X.to(device),batch_y.to(device) 
            # cleaning gradients
            optimizer.zero_grad()
            #forward pass
            logits = model(batch_X)

            #calcuate loss
            loss = criterion(logits,batch_y)

            # backprop
            loss.backward()

            #parameters update
            optimizer.step()

            total_loss = total_loss + loss.item()


        print(f"Epoch: {epoch+1} , Total_loss: {total_loss:.4f}")


    torch.save(
        {
            "iter":epochs,
            "model":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "total_loss":total_loss,
        },
        checkpoint
    )


if __name__ == "__main__":

        main()