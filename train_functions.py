import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from timeit import default_timer as timer


def train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device):
    model.train()
    train_loss, train_acc = 0,0

    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item() # accumulate train loss

        optimizer.zero_grad()

        loss.backward() # backprop

        optimizer.step() #grad descent

        # get pred probability, then find index of highest prob.
        y_pred_label = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_label == y).sum().item()/len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                device: torch.device):
  model.eval()
  test_loss, test_acc = 0,0

  with torch.inference_mode():
    for X,y in dataloader:
        X,y = X.to(device), y.to(device)

        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        test_loss += loss

        test_pred_labels = y_pred.argmax(dim=1)
        test_acc += (test_pred_labels == y).sum().item()/len(test_pred_labels)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train_model(model: torch.nn.Module, 
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                epochs: int,
                device: torch.device) -> Dict[str, List]:
      
    results = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": []
    }
    
    for epoch in tqdm(range(epochs)):
        print(f"\nEpoch: {epoch}\n--------------------")

        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=train_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        print(f"Avg Train Loss: {train_loss:.3f} | Avg Train Acc: {train_acc:.3f}%")
        print(f"Avg Test Loss: {test_loss:.3f} | Avg Test Acc: {test_acc:.3f}%")

        results["train_loss"].append(train_loss)
        results["train_accuracy"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_accuracy"].append(test_acc)
    return results