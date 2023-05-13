import matplotlib.pyplot as plt
import numpy as np
import math
from functools import reduce
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import ttest_ind
from scipy.stats import pearsonr

fname_train = 'train_ki_ref_out_train_relu_batch_32_2_lr_2_1k_adamw.txt'
fname_val = 'train_ki_ref_out_val_relu_batch_32_2_lr_2_1k_adamw.txt'
#fname_train = 'train_last_1.txt'
#fname_val = 'val_last_1.txt'

epochs = [1,50,100,150]
#epochs = [340,345,350,355,36,380]
#epochs = [400,42,440,460,480,500,550,600]
#epochs = [100]
with open(fname_train,'r') as f:
    data_train = f.readlines()  
    data_train = [eval('['+l+']') for l in data_train]
print(len(data_train))

  

with open(fname_val,'r') as f:
    data_val = f.readlines()  
    data_val = [eval('['+l+']') for l in data_val]  
    print(len(data_val)) 

    

real_data_train = [x[0] for x in data_train]
#print(real_data_train)

predicted_data_train = [x[1] for x in data_train]

real_data_val = [x[0] for x in data_val]

predicted_data_val = [x[1] for x in data_val]

max_r2_train_epoch = max([(i,r2_score(real_data_train[i],predicted_data_train[i])) for i in range(len(real_data_train))], key=lambda x:x[1])
max_r2_val_epoch = max([(i,r2_score(real_data_val[i],predicted_data_val[i])) for i in range(len(real_data_val))],key=lambda x:x[1])

epochs+=[max_r2_train_epoch[0],max_r2_val_epoch[0]]
epochs = sorted(epochs)

real_data_train = [real_data_train[i] for i in epochs]
predicted_data_train = [predicted_data_train[i] for i in epochs]
real_data_val = [real_data_val[i] for i in epochs]
predicted_data_val = [predicted_data_val[i] for i in epochs]




x = [i for i in range(10)]
#corr_coef_train = np.corrcoef(real_data_train, predicted_data_train)[0, 1]
#corr_coef_val = np.corrcoef(real_data_val, predicted_data_val)[0, 1]

#max_val = max(predicted_data_train)

fig, axs = plt.subplots(nrows=math.ceil(len(epochs)/2), ncols=2, figsize=(13, 32))

real_data_train_ext = list(reduce(lambda x,y:x+y, real_data_train))
predicted_data_train_ext = list(reduce(lambda x,y:x+y, predicted_data_train))
real_data_val_ext = list(reduce(lambda x,y:x+y, real_data_val))
predicted_data_val_ext = list(reduce(lambda x,y:x+y, predicted_data_val))

max_val = math.ceil(max([max(real_data_train_ext), max(predicted_data_train_ext), 
                  max(real_data_val_ext), max(predicted_data_val_ext)]) + 2)

min_val = math.floor(min([min(real_data_train_ext), min(predicted_data_train_ext), 
                  min(real_data_val_ext), min(predicted_data_val_ext)]))

#print(min_val)




for i, epoch in enumerate(epochs):
    #print(i)
    
    row_i,col_i = i//2,i%2
    
    # Get the data for the current epoch
    real_data_train_epoch = real_data_train[i]
    predicted_data_train_epoch = predicted_data_train[i]
    #predicted_data_train_epoch = real_data_train[i]
    real_data_val_epoch = real_data_val[i]
    predicted_data_val_epoch = predicted_data_val[i]
    #predicted_data_val_epoch = real_data_val[i]
    # Plot the four scatter plots
    axs[row_i][col_i].scatter(real_data_train_epoch, predicted_data_train_epoch, color='green',label = "Train")

    axs[row_i][col_i].scatter(real_data_val_epoch, predicted_data_val_epoch, color='blue',label = "Validation")

    axs[row_i][col_i].plot(list(range(min(0,min_val),max_val)),list(range(min(0,min_val),max_val)),color='red')

    axs[row_i][col_i].set_title("Epoch {}".format(epoch))
    axs[row_i][col_i].set_xlim(min(0,min_val),max_val)
    axs[row_i][col_i].set_ylim(min(0,min_val),max_val)
    # Add x and y labels to the leftmost and bottommost subplots
  
    axs[row_i][col_i].set_xlabel("True Value")
    axs[row_i][col_i].set_ylabel("Predicted Value")
    axs[row_i][col_i].legend()

    # Calculate RMSE for training and validation data
    #rmse_train = np.sqrt(mean_squared_error(real_data_train_epoch, predicted_data_train_epoch))
    #rmse_val = np.sqrt(mean_squared_error(real_data_val_epoch, predicted_data_val_epoch))
    # Calculate R2 score for current epoch
    diff = (a-b for a,b in zip(real_data_train_epoch , predicted_data_train_epoch))
    print([x for x in predicted_data_train_epoch if x >=15])
    print([x for x in real_data_train_epoch if x >=15])
    print([x for x in predicted_data_val_epoch if x >=15])
    print([x for x in real_data_val_epoch if x >=15])
    r2_train = r2_score(real_data_train_epoch, predicted_data_train_epoch)
    r2_val = r2_score(real_data_val_epoch, predicted_data_val_epoch)
    #print(type(r2_train))
    
    #len(r2_train)
    print("r2_train_old",r2_train)
    n_train = len(real_data_train_epoch)
    n_val = len(real_data_val_epoch)
    p = 1  # number of predictor variables
    adj_r2_train = 1 - ((1 - r2_train) * (n_train - 1) / (n_train - p - 1))
    adj_r2_val = 1 - ((1 - r2_val) * (n_val - 1) / (n_val - p - 1))
    corr_train, pval_train = pearsonr(real_data_train_epoch, predicted_data_train_epoch)
    
    # Calculate Pearson correlation coefficient and p-value for validation data
    corr_val, pval_val = pearsonr(real_data_val_epoch, predicted_data_val_epoch)
    #print(adj_r2_train)
    #print(adj_r2_val)
    #print(r2_val)
    # Plot the four scatter plots
    #axs[row_i][col_i].scatter(real_data_train_epoch, predicted_data_train_epoch, color='green',label = "Train", alpha = 0.1)

    #axs[row_i][col_i].scatter(real_data_val_epoch, predicted_data_val_epoch, color = "blue")
    axs[row_i][col_i].text(0.4, 0.85, "R2 Train: {:.2f}, p_Train: {:.2f} ".format(r2_train,corr_train), transform=axs[row_i][col_i].transAxes, fontsize=8)
    axs[row_i][col_i].text(0.4, 0.90, "R2 Validation: {:.2f}, p_Val :{:.2f} ".format(r2_val,corr_val), transform=axs[row_i][col_i].transAxes, fontsize=8)
    SSR_train= sum([(real_data_train_epoch[i]-predicted_data_train_epoch[i])**2 for i in range(len(predicted_data_train_epoch))])
    SST_train= sum([ (real_data_train_epoch[i] -np.mean(real_data_train_epoch))**2 for i in range(len(real_data_train_epoch))])
    SSR_val = sum([(real_data_val_epoch[i]-predicted_data_val_epoch[i])**2 for i in range(len(predicted_data_val_epoch))])
    SST_val = sum([ (real_data_val_epoch[i] -np.mean(real_data_val_epoch))**2 for i in range(len(real_data_val_epoch))])
 
    #print('########## Epoch {} ##########\n,\tSSR_train = {} \n\tSST_train = {} \n\tR2_train = {}\n\tSSR_val = {}\n\tSST_val = {}\n\tR2_val = {}\n'.format(epoch,SSR_train,SST_train,1.0-(SSR_train/SST_train),SSR_val,SST_val,1.0-(SSR_val/SST_val)))
    t_stat, p_val = ttest_ind(real_data_train_epoch, predicted_data_train_epoch)
    #print('p-value, eoich:', p_val,epoch)
    
    #print("r2_cl",  r2_train_cl)
    '''
    # Print the correlation coefficients and p-values for the current epoch
    print(f"Epoch {epoch}")
    print(f"Training data: correlation coefficient = {corr_train:.4f}, p-value = {pval_train:.4f}")
    print(f"Validation data: correlation coefficient = {corr_val:.4f}, p-value = {pval_val:.4f}")
    '''
    #axs[row_i][col_i].set_title("Epoch {} (Train RMSE: {:.4f}, Val RMSE: {:.4f})".format(epoch, rmse_train, rmse_val))
    #axs[row_i][col_i].set_title("Epoch {}\nTrain R2: {:.3f}\nVal R2: {:.3f}".format(epoch, r2_train, r2_val))
    axs[row_i][col_i].set_xlim(min(0,min_val),max_val)
    axs[row_i][col_i].set_ylim(min(0,min_val),max_val)
    axs[row_i][col_i].legend()

# Add a title to the figure
fig.suptitle("Train and Validation Set Predictions \n Optimizer: Adamw \n Trained for 1000 epochs")

# Save the figure
plt.savefig("train_val_prediction_relu_batch_32_2_lr_2_500_adamw_11.png")





