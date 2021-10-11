from typing import *

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class Trainer():
    """Utility class to train and evaluate a model."""

    def __init__(self, model, loss_function, optimizer, device, 
                 include_chars=False, include_pos=False,
                 verbose_level=2, verbose_interval=100):
        """
        Args:
            model:              The model we want to train.
            loss_function:      The loss function to minimize.
            optimizer:          The optimizer used to minimize loss_function.
            device:             The device to train on.
            include_chars:      Whether to include character-level embeddings. 
            include_pos:        Whether to include POS embeddings.     
            verbose_level:      The level of verbosity for logging progress.
            verbose_interval:   The number of steps between logs.
        """
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        self.include_chars = include_chars
        self.include_pos = include_pos
        self.verbose_level = verbose_level
        self.verbose_interval = verbose_interval

        self.model.to(self.device)  # move model to GPU if available

    def train(self, train_dataset:DataLoader, dev_dataset:DataLoader, epochs:int=1,
              early_stopping=True, min_delta=1e-9) -> float:
        """
        Args:
            train_dataset:  A Dataset or DataLoader instance containing
                            the training instances.
            dev_dataset:    A Dataset or DataLoader instance used to evaluate
                            learning progress.
            epochs:         The number of times to iterate over train_dataset.

        Returns:
            avg_train_loss: the average training loss on train_dataset over
                epochs.
        """
        
        assert epochs > 1 and isinstance(epochs, int)
        if self.verbose_level > 0:
            print('Training...')
        
        train_loss = 0.0
        n_batches = len(train_dataset)

        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                self.model.train()
                
                batch_iterator = enumerate(train_dataset)
                if self.verbose_level > 0:
                    # track progress via tqdm
                    batch_iterator = tqdm(batch_iterator, total=n_batches, 
                            desc="Train Epoch {:d}/{:d}".format(epoch+1, epochs))
                
                for step, sample in batch_iterator:
                    x = sample['tokens'].to(self.device)
                    y_true = sample['labels'].to(self.device)

                    self.optimizer.zero_grad()
                    
                    x_args = [x]
                    if self.include_chars:
                        x_chars = sample["chars"].to(self.device)
                        x_args.append(x_chars)
                    
                    if self.include_pos:
                        x_pos = sample["pos_tags"].to(self.device)
                        x_args.append(x_pos)
                        
                    y_pred = self.model(*x_args)
                    
                    y_pred = y_pred.view(-1, y_pred.shape[-1])
                    y_true = y_true.view(-1)
                    sample_loss = self.loss_function(y_pred, y_true)

                    sample_loss.backward()
                    self.optimizer.step()

                    epoch_loss += sample_loss.detach().cpu().item()

                    if self.verbose_level > 1 and step > 0 and step % self.verbose_interval == 0:
                        avg_loss = epoch_loss / (step+1)
                        # log progress
                        batch_iterator.set_postfix_str("avg loss={:.4f}".format(avg_loss))
                
                avg_loss = epoch_loss / n_batches
                train_loss += avg_loss

                dev_loss = self.evaluate(dev_dataset)

                if epoch >= 1 and early_stopping and \
                    old_dev_loss - dev_loss <= min_delta:
                    print("Training interrupted by early stopping.")
                    break
                
                old_dev_loss = dev_loss
        
        except KeyboardInterrupt:
            print("Training interrupted by user.")
        
        if self.verbose_level > 0:
            print('Done!')
        
        avg_loss = train_loss / epoch
        return avg_loss
    
    def evaluate(self, dev_dataset: DataLoader) -> float:

        """
        Args:
            dev_dataset: The dataset to use to evaluate the model.

        Returns:
            The average loss over dev_dataset.
        """
        assert all([param is not None for param in [
                self.device, self.verbose_level, self.verbose_interval]])
        
        dev_loss = 0.0
        self.model.eval()

        n_batches = len(dev_dataset)

        with torch.no_grad():
            batch_iterator = enumerate(dev_dataset)
            if self.verbose_level > 0:
                batch_iterator = tqdm(batch_iterator, total=n_batches, desc="Dev Epoch")
                
            for step, sample in batch_iterator:
                x = sample['tokens'].to(self.device)
                y_true = sample['labels'].to(self.device)
                
                x_args = [x]
                if self.include_chars:
                    x_chars = sample["chars"].to(self.device)
                    x_args.append(x_chars)
                
                if self.include_pos:
                    x_pos = sample["pos_tags"].to(self.device)
                    x_args.append(x_pos)
                    
                y_pred = self.model(*x_args)
                
                y_pred = y_pred.view(-1, y_pred.shape[-1])
                y_true = y_true.view(-1)
                sample_loss = self.loss_function(y_pred, y_true)

                dev_loss += sample_loss.detach().cpu().item()

                if self.verbose_level > 1 and step > 0 and step % self.verbose_interval == 0:
                    avg_loss = dev_loss / step
                    batch_iterator.set_postfix_str("avg loss={:.4f}".format(avg_loss))
        
        return dev_loss / len(dev_dataset)


def train(
    model: nn.Module, 
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    train_dataloader: DataLoader,
    dev_dataloader: DataLoader,
    n_epochs: int,
    device: str = DEVICE,
    include_chars=True,
    include_pos=True,
    verbose_level=2,
    verbose_interval=10
    ):
    
    
    trainer = Trainer(model=model, 
                    loss_function=loss_function, 
                    optimizer=optimizer, 
                    device=device, 
                    include_chars=include_chars,
                    include_pos=include_pos,
                    verbose_level=verbose_level,
                    verbose_interval=verbose_interval
                    )

    avg_train_loss = trainer.train(train_dataloader, dev_dataloader, n_epochs)
    print("Avg training loss:", avg_train_loss)