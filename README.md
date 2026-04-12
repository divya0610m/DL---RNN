# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.

The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

<img width="671" height="840" alt="image" src="https://github.com/user-attachments/assets/1ac46094-9fe1-4187-a52b-397764bdef57" />

<img width="672" height="757" alt="image" src="https://github.com/user-attachments/assets/051b9b7c-fefa-488b-8981-538492c73625" />


## DESIGN STEPS

### STEP 1: 

Load and normalize data, create sequences.

### STEP 2: 

Convert data to tensors and set up DataLoader.

### STEP 3: 

Define the RNN model architecture

### STEP 4: 

Summarize, compile with loss and optimizer.

### STEP 5: 

Train the model with loss tracking.

### STEP 6: 

Predict on test data, plot actual vs. predicted prices.

## PROGRAM

### Name: DIVYA LAKSHMI M 

### Register Number: 212224040082

```python

## Step 2: Define RNN Model
class RNNModel(nn.Module):
  def __init__(self,input_size=1,hidden_size=64,num_layers=2,output_size=1):
    super(RNNModel,self).__init__()
    self.rnn=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
    self.fc=nn.Linear(hidden_size,output_size)
  def forward(self,x):
    out,_=self.rnn(x)
    out=self.fc(out[:,-1,:])
    return out

model = RNNModel()
criterion =nn.MSELoss()
optimizer =torch.optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
      total_loss=0
      for x_batch,y_batch in train_loader:
        x_batch,y_batch=x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        outputs=model(x_batch)
        loss=criterion(outputs,y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
      train_losses.append(total_loss/len(train_loader))
      print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
     
train_model(model,train_loader,criterion,optimizer)


```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="702" height="802" alt="Screenshot 2026-03-09 091523" src="https://github.com/user-attachments/assets/2dbdc822-40d5-4a9a-aa4f-d04ac8897e80" />

## True Stock Price, Predicted Stock Price vs time

<img width="1092" height="659" alt="Screenshot 2026-03-09 091716" src="https://github.com/user-attachments/assets/a88d381e-73b8-46ef-bc62-1f9a9d6cc1c1" />

### Predictions

<img width="316" height="52" alt="image" src="https://github.com/user-attachments/assets/787b5dfd-f421-467c-9550-14843f2b0e94" />

## RESULT

Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.


