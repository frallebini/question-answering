from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from config import conf
from tqdm import tqdm
from  transformers import DistilBertForQuestionAnswering


def train_loop(model: DistilBertForQuestionAnswering, 
               train_loader: DataLoader, 
               val_loader: DataLoader,
               opt: torch.optim.Optimizer,
               device: torch.device) -> tuple[float, float, float, float]:

    for epoch in range(conf['N_EPOCHS']):
        model.train()

        n_train = 0
        train_loss = 0
        train_acc = 0

        train_iter = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for train_batch in train_iter:
            input_ids = train_batch['input_ids'].to(device)
            attention_mask = train_batch['attention_mask'].to(device)
            start_true = train_batch['start_positions'].to(device)
            end_true = train_batch['end_positions'].to(device)

            outputs = model(input_ids, 
                            attention_mask=attention_mask,
                            start_positions=start_true,
                            end_positions=end_true)
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)

            assert len(start_true) == \
                len(start_pred) == \
                len(end_true) == \
                len(end_pred)
            n_samples = len(start_true)

            loss = outputs['loss']  # computed as (start_loss + end_loss) / 2
            # where start_loss and end_loss are, in turn, computed as averages 
            # over the batch
            start_acc = compute_accuracy(start_pred, start_true)
            end_acc = compute_accuracy(end_pred, end_true)
            total_acc = (start_acc + end_acc) / 2

            train_iter.set_postfix(loss=loss.item(), acc=total_acc)
            
            n_train += n_samples
            train_loss += loss.item() * n_samples
            train_acc += total_acc * n_samples
            
            loss.backward()
            
            opt.step()
            opt.zero_grad()
        
        n_val, val_loss, val_acc = evaluate(model, val_loader, device)
        
        train_loss /= n_train
        train_acc /= n_train
        val_loss /= n_val
        val_acc /= n_val

        print(f'Epoch {epoch}: '           +
              f'loss={train_loss:.3f}, '   + 
              f'acc={train_acc:.3f}, '     +
              f'val_loss={val_loss:.3f}, ' +
              f'val_acc={val_acc:.3f}')
        
    return train_loss, train_acc, val_loss, val_acc


def evaluate(model: DistilBertForQuestionAnswering,
             val_loader: DataLoader,
             device: torch.device) -> tuple[int, float, float]:
    
    model.eval()
            
    n_val = 0
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for val_batch in val_loader:
            input_ids = val_batch['input_ids'].to(device)
            attention_mask = val_batch['attention_mask'].to(device)
            start_true = val_batch['start_positions'].to(device)
            end_true = val_batch['end_positions'].to(device)
            
            outputs = model(input_ids, 
                            attention_mask=attention_mask,
                            start_positions=start_true,
                            end_positions=end_true)
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)

            assert len(start_true) == \
                   len(start_pred) == \
                   len(end_true) == \
                   len(end_pred)
            n_samples = len(start_true)

            loss = outputs['loss'].item()  # computed as 
            # (start_loss + end_loss) / 2
            # where start_loss and end_loss are, in turn, computed as averages 
            # over the batch
            start_acc = compute_accuracy(start_pred, start_true)
            end_acc = compute_accuracy(end_pred, end_true)
            total_acc = (start_acc + end_acc) / 2

            n_val += n_samples
            val_loss += loss * n_samples
            val_acc += total_acc * n_samples

            return n_val, val_loss, val_acc
    

def compute_accuracy(pred: torch.Tensor, true: torch.Tensor) -> float:
    assert len(pred) == len(true)
    return ((pred == true).sum() / len(pred)).item()
