import matplotlib.pyplot as plt
import numpy as np
import math
from functools import reduce
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
fname_train = 'train_ki_ref_out_train_relu_batch_32_2_lr_2_1k_adamw.txt'
fname_val = 'train_ki_ref_out_val_relu_batch_32_2_lr_2_1k_adamw.txt'

def find_outliers(data):
    # Calculate the first and third quartiles
    q1, q3 = np.percentile(data, [25, 75])
    # Calculate the interquartile range
    iqr = q3 - q1
    # Define the lower and upper bounds for outliers
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    # Find the indices of the outliers
    outliers = np.where((data < lower_bound) | (data > upper_bound))
    # Print the values of the outliers
    return  outliers[0]

    
def remove_outliers(data):
    # Calculate the first and third quartiles
    q1, q3 = np.percentile(data, [25, 75])
    # Calculate the interquartile range
    iqr = q3 - q1
    # Define the lower and upper bounds for outliers
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    # Remove the outliers from the data
    data_clean = np.where((data >= lower_bound) | (data <= upper_bound))
    return data_clean

epochs = [1000]
with open(fname_train,'r') as f:
    data_train = f.readlines()  
    data_train = [eval('['+l+']') for l in data_train]
  

with open(fname_val,'r') as f:
    data_val = f.readlines()  
    data_val = [eval('['+l+']') for l in data_val]   

real_data_train = [x[0] for x in data_train]
#real_data_train = np.sqrt(real_data_train)
predicted_data_train = [x[1] for x in data_train]
#predicted_data_train = np.sqrt(predicted_data_train)

real_data_val = [x[0] for x in data_val]
#real_data_val= np.sqrt(real_data_val)
predicted_data_val = [x[1] for x in data_val]
#predicted_data_val= np.sqrt(predicted_data_val)

real_data_train = [real_data_train[i] for i in epochs]
predicted_data_train = [predicted_data_train[i] for i in epochs]
real_data_val = [real_data_val[i] for i in epochs]
predicted_data_val = [predicted_data_val[i] for i in epochs]

x = [i for i in range(10)]
corr_coef_train = np.corrcoef(real_data_train, predicted_data_train)[0, 1]
corr_coef_val = np.corrcoef(real_data_val, predicted_data_val)[0, 1]


real_data_train_ext = list(reduce(lambda x,y:x+y, real_data_train))
predicted_data_train_ext = list(reduce(lambda x,y:x+y, predicted_data_train))
real_data_val_ext = list(reduce(lambda x,y:x+y, real_data_val))
predicted_data_val_ext = list(reduce(lambda x,y:x+y, predicted_data_val))

max_val = math.ceil(max([max(real_data_train_ext), max(predicted_data_train_ext), 
                  max(real_data_val_ext), max(predicted_data_val_ext)]) + 2)

min_val = math.floor(min([min(real_data_train_ext), min(predicted_data_train_ext), 
                  min(real_data_val_ext), min(predicted_data_val_ext)]))

print(min_val)

real_data_train_clean = []
real_data_val_clean = []
#for i, epoch in enumerate(epochs):
for i in range(len(epochs)):
    
    # Get the data for the current epoch
    real_data_train_epoch = real_data_train[i]
    predicted_data_train_epoch = predicted_data_train[i]
    real_data_val_epoch = real_data_val[i]
    predicted_data_val_epoch = predicted_data_val[i]

    # Plot the four scatter plots
    plt.figure(figsize=(20, 20))
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 12))

    plt.title("Epoch BoxPlot {}".format(epochs))
    plt.xlim(min(0,min_val),max_val)
    plt.ylim(min(0,min_val),max_val)
    # Add x and y labels to the leftmost and bottommost subplots
  
    #plt.xlabel("True Value")
    #plt.ylabel("Predicted Value")
    #plt.legend()

    # Calculate RMSE for training and validation data
    rmse_train = np.sqrt(mean_squared_error(real_data_train_epoch, predicted_data_train_epoch))
    rmse_val = np.sqrt(mean_squared_error(real_data_val_epoch, predicted_data_val_epoch))
    # Calculate R2 score for current epoch
    r2_train = r2_score(real_data_train_epoch, predicted_data_train_epoch)
    r2_val = r2_score(real_data_val_epoch, predicted_data_val_epoch)
    print("r2 train: ",r2_train)
    train_outliers = find_outliers(real_data_train_epoch)
    print("Epoch", i, "Training Data Outliers:",train_outliers)

    print("Epoch", i, "Training Data Outliers:", [(pos,real_data_train_epoch[pos]) for pos in train_outliers])
    
    # Find outliers in the validation data
    val_outliers = find_outliers(real_data_val_epoch)
    print("Epoch", i, "Validation Data Outliers:", val_outliers)
    data_clean= remove_outliers(real_data_train_epoch)
    print("Cleaned data:", data_clean)
    #print("Outliers:", outliers)
    #plt.text(15, 2, "R2 Train: {:.2f}".format(r2_train), fontsize=12)
    #plt.text(15, 0.08, "R2 Validation: {:.2f}".format(r2_val), fontsize=12,horizontalalignment='left',verticalalignment='bottom')
    real_data_train_epoch_clean = remove_outliers(train_outliers)
    real_data_train_clean.append(real_data_train_epoch_clean)
    real_data_train_epoch_clean=real_data_train_epoch_clean[i]
    print(list(real_data_train_clean))
    
    # Remove outliers from the validation data
    real_data_val_epoch_clean = remove_outliers(real_data_val_epoch)
    
    real_data_val_clean.append(real_data_val_epoch_clean)
    
    plt.xlim(min(0,min_val),max_val)
    plt.ylim(min(0,min_val),max_val)
    #sns.regplot(x=real_data_train_epoch,y= predicted_data_train_epoch,ci=95,label = "Train",scatter_kws={"color": "lightskyblue"})

    #sns.regplot(x=real_data_val_epoch,y= predicted_data_val_epoch, ci=95,scatter_kws={"color": "lightgreen"},line_kws={"color": "green"},label = "Validation")

    plt.legend()
    sns.boxplot(x=predicted_data_train_epoch,ax=axs[0,0])
    sns.boxplot(x=predicted_data_val_epoch,ax=axs[0,1])
    sns.boxplot(x=real_data_train_epoch,ax=axs[1,0])
    sns.boxplot(x=real_data_val_epoch,ax=axs[1,1])
    axs[0,0].set_title('Train predicted values')
    axs[0,1].set_title('Validation predicted values')
    axs[1,0].set_title('Train real values')
    axs[1,1].set_title('Validation real values')
    
   


fig.suptitle("Epoch 100")
plt.savefig("train_val_predictions_relu_batch_32_r2_1000_adamw_cl.png")



