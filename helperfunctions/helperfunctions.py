from typing import Dict, List
import pandas
import matplotlib.pyplot as plt
import torch

# Defining the plot_loss_curve function for Evaluation

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """

    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

# Defining the table_to_matrix function

def table2matrix( table: pandas.DataFrame,
                 normalization: str = "minmax",
                 prescaled: str = "no",
                 digits : int = 0.1):
  """
  This table2matrix is custom built for the Breast Cancer Wisconsin (Diagnostic) Data Set and needs to be adjusted accordingly.
  table2matrix converts a csv table (1-D) into matrices (2-D) such that ML architectures such as ResNet and CNN can be applied on the dataset.
  After conversion it is recommended to apply a custom Dataset and a custom DataLoader seperate from the tabular data. Labels are preservered while the features are arranged as
  matrices according to their values, normalized values across all features.

  Steps:

  1. Normalize the features

  2.Each feature of each row is converted into a normalized value between [0:1]
  According to this value (f.e 0.7) this feature is then transformed into a  10-D column where the first 7 entries are 1 and the last 3 are 0.

  3. Those Columns are then added up to a matrix


  This function is custom built for the Breast Cancer Wisconsin (Diagnostic) Data Set and needs to be adjusted accordingly.
  Args:
      table[pandas.Dataframe]: the columns mark a certain type of feature and the rows are individual entries
      normalization[Str] : Determines whether to use "mean" or "minmax" normalization, defaults to Gaussian normalization
      prescale[Str] : For datasets that are already scaled or are easily scaled via other methods, prescaled = "yes" and this function skips the inbuilt scaling steps
                      for prescaled ="no" the afforementioned normalization is applied.

  Output:
      tensor_table_matrix[nn.tensor] : Output is a
  """
  #0. Normalize the features
  if prescaled == "no":
    if normalization == "mean":
      #Applies the mean normalization (DOES NOT CONFINE the output space to [0:1])
      table = (table-table.mean())/table.std()
      #print ("[INFO]: Finshed Mean Normalization of the features across all entries!")

    elif normalization == "minmax":
      #Applies the min-max normalization
      table = (table-table.min())/(table.max()-table.min())
      #print ("[INFO]: Finshed Minmaxed Normalization of the features across all entries!")
  else:
    print ("[INFO]: Dataframe was marked as prescaled, no further of the features across all entries applied!")

  #1.1 Create empty matrix (python list) to be filled for the whole dataset
  new_table =[]

  for i in range(table.shape[0]):
    #1.2 Create matrix for the new feature vectors
    output_matrix = []
    for j in range(table.shape[1]):
      #1.3 Transform the features to vectors
      feature2vector = []
      rounded_value = round(table.iloc[i][j]/digits)
      for i in range(0, rounded_value):
        #1.4 Fill the matrix with the vector 1
        feature2vector.append(1)

      #1.5 fill the rest of the spots with 0
      for i in range(int(1/digits)-len(feature2vector)):
        feature2vector.append(0)

      #1.6 Fill the outputmatrix with the vectors
      output_matrix.append(feature2vector)


    #print("[INFO] Finished Creating column vectors from normalized features")

    #1.7 Converting the lists of vectors[lists] into a tensor for pytorch usage
    output_matrix = torch.tensor(output_matrix, dtype = torch.float64)

    #print("[INFO] Finished converting the output_matrix into a tensor")
    new_table.append(output_matrix)



  print(f"[INFO] Finished combining vectors into matrix with how datatype {type(new_table)}")
  return new_table
