import torch
from torch import nn
from tqdm.auto import tqdm

#Define the Fully-Connected model

class FC_NN(nn.Module):

  """
  Note this model only returns the logits and might still need sigmoid
  have to be careful while training it or when pipelining it with the others
  """
  def __init__(self, n_in, n_out):
    super().__init__()
    # 2. Create nn.Linear layers capable of handling X and y input and output
    self.layer_1 = nn.Linear(in_features =n_in , out_features=n_in)
    self.layer_2 = nn.Linear(in_features = n_in, out_features = n_in)
    self.layer_3 = nn.Linear(in_features = n_in, out_features = n_out)
    self.relu = nn.ReLU()
    self.double()

  def forward(self, x):
    return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

# Define the 1-D Convolutional Neural Network

import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
  def __init__(self, n_features):
    super().__init__()
    # Create First Convolutional layer
    self.conv1 = nn.Conv1d(
        in_channels =1,
        out_channels = 32,
        kernel_size = 3,
        padding =1
    )
    self.conv2 = nn.Conv1d(
        in_channels = 32,
        out_channels= 64,
        kernel_size = 3,
        padding = 1
    )

    self.pool = nn.MaxPool1d(2)
    conv_output_size = (n_features // 2) * 64
    self.fc = nn.Linear(conv_output_size, 1)
    self.double()

  def forward(self, x):
    x = x.unsqueeze(1)
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.flatten(1)

    return self.fc(x)

class CNN(nn.Module):
  def __init__(self, n_features, n_in_channels):
    super().__init__()

    # Create First Convolutional Layer
    self.conv1 = nn.Conv2d(in_channels = n_in_channels, out_channels = 8, kernel_size =3, padding =1 )

    # Creat Pooling Layer
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    #Create Second Convolutional Layer
    self.conv2  =nn.Conv2d(in_channels = 8, out_channels =16, kernel_size = 3, padding =1)

    # Using LazyLinear that automatically inferes the number of input parameters after convolution and pooling
    self.fc1 = nn.LazyLinear(500)
    self.fc2 = nn.Linear(500,1)
    self.double()

  def forward(self, x):
    """
    Define the forward pass of this CNN Architecture

    Args:
      x: input tensor

    Returns:
      Output 1-D tensor for binary classification
    """
    print(f"x.shape before usqueezing : {x.shape}")
    x = x.unsqueeze(1)
    print(f"x.shape after unsqueezing : {x.shape}")
    x = F.relu(self.conv1(x))    # Apply first convolution and ReLU activation
    print(f"x.shape after first convolution : {x.shape}")
    x = F.relu(self.conv2(x))    # Apply second convolution and ReLU activation
    print(f"x.shape after seond convolution : {x.shape}")
    x = x.reshape(x.shape[0], -1)# Apply min pooling
    print(f"x.shape after reshaping : {x.shape}")
    x = F.relu(self.fc1(x))             # Apply fully connected layer
    x = self.fc2(x)
    return x
