import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns

fd_res = open('validation_ki_external_relu_batch_32_2_lr_2_300_adamw_3.txt','r')
fd_res_t = open('train_ki_external_relu_batch_32_2_lr_2_300_adamw_3.txt','r')
res = eval(fd_res.readlines()[0])
res_t = eval(fd_res_t.readlines()[0])
fd_res.close()
fd_res_t.close()
#train = [math.sqrt(x[1]) for x in res_t]
#val= [math.sqrt(x[1]) for x in res]
y_label = 'MAE'
train = [x[2] for x in res_t]
val = [x[2] for x in res]

#MSE
#train = [x[0] for x in res]
#val = [x[1] for x in res]
assert len(train) == len(val)

epoch = np.arange(len(train))
plt.scatter(epoch,train, label = "Train set")
plt.scatter(epoch,val, label = "Validation set")
plt.legend()
plt.title("Output channels = 32")
#plt.ylim(0,max(max(train),max(val)))
#plt.xlim(-1,100)
plt.ylabel(y_label)
plt.xlabel("Epochs")
#plt.title("Optimizer: ADAM, LR= 0.0001")
import matplotlib.pyplot as plt


# plot MSE values
#sns.kdeplot([x[1] for x in res], shade=True,label='Validation MSE')
#sns.kdeplot([x[1] for x in res], shade=True, label='Train MSE')

plt.xlabel("Epoch")
plt.ylabel(y_label)
plt.savefig("TEST_eval_train_ki_relu_{}_32_2_scatter_lr_2_300.png".format(y_label))
'''
# Identify the outlier systems at epoch 100
epoch = np.arange(len(train))

train_epoch100 = [(x[0]) for x in res]

val_epoch100 = [(x[1]) for x in res]
q1_train, q3_train = np.percentile(train_epoch100, [25, 75])
iqr_train = q3_train - q1_train
lower_train = q1_train - 1.5 * iqr_train
upper_train = q3_train + 1.5 * iqr_train
outlier_train = [x for x in train_epoch100 if x < lower_train or x > upper_train]
print("outlier train",outlier_train)
q1_val, q3_val = np.percentile(val_epoch100, [25, 75])
iqr_val = q3_val - q1_val
lower_val = q1_val - 1.5 * iqr_val
upper_val = q3_val + 1.5 * iqr_val
outlier_val = [x for x in val_epoch100 if x < lower_val or x > upper_val]
print("outlier val",outlier_val)
print("TRAIN LOWER, TRAIN UPPER ",(lower_train,upper_train))
print("VAL LOWER, VAL UPPER ",(lower_val,upper_val))


# Create the box plot
fig, ax = plt.subplots()
ax.boxplot([outlier_train, outlier_val], labels=['Training', 'Validation'])
ax.set_title('Box Plot of Outliers at Epoch 100')
ax.set_ylabel('Square Root of Evaluation Metric')
plt.savefig("TEST_eval_train_ki_relu_batch_32_lr.png")




# Print the outliers


'''

