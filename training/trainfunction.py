import torch
from tqdm.auto import tqdm
def train_step(device,
              model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module,
               optimizer: torch.optim.Optimizer):

  """
  Custom train_step function that performs forward pass and backpropagation as well as keep track of the loss and accuracy
  NOTE: the loss fn should have a sigmoid since pure logits are send to the loss_fn
  """
  # Put model in train mode
  model.train()

  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
    #Send data to target device
    X, y = X.to(device), y.to(device)

    #1. Forward pass
    y_pred = model(X)

    #Transform the logits into probabilities
    y_pred_prob = torch.sigmoid(y_pred)

    #2. Calculate the accumulate loss
    loss = loss_fn(y_pred_prob, y.unsqueeze(1).double())
    train_loss += loss.item()

    #3. Optimizer zero grad
    optimizer.zero_grad()

    #4. Loss backward
    loss.backward()

    #5. optimizer Step()
    optimizer.step()

    #Calculate and accumulate accuracy metrics across all batches
    y_pred_class = torch.round(y_pred_prob)

    train_acc += torch.eq(y_pred_class, y.unsqueeze(1).double()).sum().item()/len(y_pred_class)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)

  return train_loss, train_acc





def test_step(device,
              model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module):

  """
  Custom test_step function that performs forward pass and backpropagation as well as keep track of the loss and accuracy
  NOTE: the loss fn should have a sigmoid since pure logits are send to the loss_fn
  """
  # Put model in train mode
  model.eval()

  # Setup train loss and train accuracy values
  test_loss, test_acc = 0, 0

  with torch.inference_mode():
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
      #Send data to target device
      X, y = X.to(device), y.to(device)

      #1. Forward pass
      test_pred_logits = model(X)
      test_pred_probs = torch.sigmoid(test_pred_logits)


      #2. Calculate the accumulate loss
      loss = loss_fn(test_pred_probs, y.unsqueeze(1).double())
      test_loss += loss.item()

      #Calculate and accumulate accuracy metrics across all batches

      test_pred_class = torch.round(test_pred_probs)
      test_acc += torch.eq(test_pred_class, y.unsqueeze(1).double()).sum().item()/len(test_pred_class)

  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)

  return test_loss, test_acc

def train (device,
          model: torch.nn.Module,
           train_dataloader: torch.utils.data.DataLoader,
           test_dataloader: torch.utils.data.DataLoader,
           optimizer: torch.optim.Optimizer,
           loss_fn: torch.nn.Module,
           epochs: int = 100):

  # 2. Create an empty results dictionary
  results = {
      "train_loss" : [],
      "train_acc" : [],
      "test_loss" : [],
      "test_acc" : []
  }

  # 3. Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step( device = device,
                                      model = model,
                                      dataloader = train_dataloader,
                                      loss_fn = loss_fn,
                                      optimizer = optimizer)
    test_loss, test_acc = test_step(device = device,
                                    model = model,
                                    dataloader = test_dataloader,
                                    loss_fn = loss_fn)

    # 4. Print out what's happening
    if (epoch% 10==0):
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss : {train_loss:.4f} | "
          f"test_loss : {test_loss:.4f} | "
          f"test_acc  :  {test_acc:.4f} | "
          f"train_acc : {train_acc:.4f} | "
      )

    # 5. Update results dictionary
    # Ensure all data is moved to CPU and converted to float for storage
    results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
    results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
    results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
    results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    #6. Return the filled results at the end of the epochs
  return results
