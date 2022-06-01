'''
Prepared by Madeleine D. Breshears (mdbresh@uw.edu)

Corresponding author: David S. Ginger (dginger@uw.edu)

Prepared for publication of corresponding paper title:
A Robust Neural Network for Extracting Dynamics from Time-Resolved Electrostatic
Force Microscopy Data

June 1, 2022
'''

import numpy as np
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class Network(pl.LightningModule):
    def __init__(self, classes=10):
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(100,100), #0
            torch.nn.ReLU(), #1
            torch.nn.Dropout(0.1),#2
            torch.nn.Linear(100,100),#3
            torch.nn.ReLU(), #4
            torch.nn.Linear(100,10), #5
            torch.nn.ReLU(),#6
            torch.nn.Linear(10,1) #7
            )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y) # L1 loss measures absolute distance to target
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'tensorboard_logs': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss)
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.l1_loss(y_hat, y)
        return {'test loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

def train_model(model, data, path, epochs=200, gpu=True):
    '''
    Train a model with your own data for a given number of epochs either with
    or without a GPU. Without a GPU the training process can take up a lot of
    RAM and/or crash. It is recommended that a GPU is used.

    Inputs --

    model : PyTorch Lightning model. Specifically the Network() class shown
        in this file.

    data : Numpy array of dimensions num_samples by 101, where data[:,0] contains
        the targets or labels for the data, data[:,1] contains the k (N/m)
        values for the cantilever(s), data[:,2] contains the Q factors for the
        cantilever(s), and data[:,3:] contains the instantaneous frequency
        signals cropped from 50% of time before the trigger (0.2 ms) and 90%
        through the experiment (1.8 ms), resampled to be 98 indices long, and
        normalized between 0 and 1.

    path : String describing the path to save resulting data loaders and trained
        network checkpoints.

    epochs : Integer value of how many epochs to train for.

    gpu : Boolean regarding whether or not to use GPU resources.

    Returns -- nothing, but saves model checkpoints at each epoch to "path"
        directory. Also saves the training, validation, and testing data loaders.
    '''

    # Define model checkpoints to save model weights for each epoch
    checkpoint_callback = ModelCheckpoint(save_top_k=-1, filename=path+'epoch:02d')

    # if GPU is available, use it!
    if gpu==True:
        trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=epochs,
        callbacks=[checkpoint_callback], gpus=1)

    # Warning: if GPU is not available, code may crash due to RAM limitations.
    # GPU is recommended but if that's not possible train for fewer epochs (~20).
    # This will result in poor performing model but it will work!
    else:
        trainer = Trainer(progress_bar_refresh_rate=20, max_epochs=epochs,
        callbacks=[checkpoint_callback])

    # Grab our targets and our input data.
    y = np.stack(data[:,0])
    x = np.stack(data[:,1:])

    # Convert to tensors of type Float.
    tensor_x = torch.from_numpy(x).float()
    tensor_y = torch.from_numpy(y).float().unsqueeze(1)

    # Add to tensor dataset to be sorted.
    dataset = TensorDataset(tensor_x,tensor_y)

    # Default train:val:test ratio is 70:15:15
    train = int(0.7*y.shape[0])
    val = int(0.15*y.shape[0])
    test = y.shape[0] - train - val

    # Randomly split dataset according to train:val:test ratio
    train_set, val_set, test_set = torch.utils.data.random_split(dataset,
    [train, val, test])

    # Load in DataLoaders for Trainer handling.
    # Manually decrease num_workers if computer is too taxed.
    train_load = DataLoader(train_set, num_workers=5)
    val_load = DataLoader(val_set, num_workers=5)
    test_load = DataLoader(test_set, num_workers=5)

    # Train model.
    trainer.fit(model, train_load, val_load)

    # Save final model checkpoint.
    trainer.save_checkpoint(path+'trained_network.ckpt')

    # Save data loaders.
    # Data loaders can be loaded again using the torch.load() function.
    torch.save(train_load, path+'training_data.pt')
    torch.save(val_load, path+'validation_data.pt')
    torch.save(test_load, path+'test_data.pt')

def extract_tau(model, data, num=100, dists=True, im=True):
    '''
    Function extracts tau from data according to the quasi-ensembling technique
    discussed in the paper text.

    Inputs --

    model : PyTorch Lightning model. Specifically the Network() class shown
        in this file.

    data : Numpy array where the -1 dimension is 100. If the data is a single
        point scan or [k, q, omega(t)] trace, then data.shape should be
        [1,100], where [0,0] is the k (N/m) value for the cantilever, [0,1] is
        the Q factor for the cantilever, and [0,2:] is the instantaneous
        frequency signal cropped from 50% before the trigger (0.2 ms) to 90%
        of the total experiment (1.8 ms), resampled to be 98 indices long, and
        normalized from 0 to 1. If the data is an image of shape [256,256,100]
        (or whatever resolution your image is), then [:,:,0] is k (N/m), [:,:,1]
        is Q, and [:,:,2:] is the prepared instantaneous frequency traces.

    num : Integer number of predictions to calculate before averaging and
        calculating the standard deviation according to the quasi-ensembling
        method proposed by Gal et al.
        (DOI: https://doi.org/10.48550/arXiv.1506.02142)

    dists : Boolean that determines whether or not to return the list of
        predictions or just the final average prediction (with the standard
        deviation).

    im : Boolean that determines if the input data is an image (3 dimensional
        numpy array) or not (2 dimensional numpy array).

    Returns --

    pred : Numpy array of either the list of prediction(s) or the image.

    std : Standard deviation.

    preds : (only if dist == True) List of num predictions before averaging
        to obtain pred and std.
    '''
    preds = []
    if im==True:
        for i in range(num):
            # Make num predictions
            p = model(torch.from_numpy(data).float()).detach().numpy()[:,:,0]
            preds.append(p)
        # Average the predictions and calculate the std.
        pred = np.mean(np.stack(preds),axis=0)
        std = np.std(np.stack(preds),axis=0)
    else:
        for i in range(num):
            # Make num predictions
            p = model(torch.from_numpy(data).float()).detach().numpy()[:,0]
            preds.append(p)
        # Average the predictions and calculate std.
        pred = np.mean(np.stack(preds),axis=0)
        std = np.std(np.stack(preds),axis=0)

    if dists==True:
        return pred, std, np.stack(preds)
    else:
        return pred, std
