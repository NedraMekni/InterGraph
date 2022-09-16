import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


if __name__=='__main__':

    warnings.filterwarnings('ignore')
    #df = pd.read_csv("/home/nmekni/Documents/NLRP3/InterGraph/scripts/test_y.csv")
    #print(df.head())
    df = pd.read_csv("/home/nmekni/Documents/NLRP3/InterGraph/data/csv/data.csv",usecols =[' pdb_code', ' activity'])
    
    df = df.mask(df.eq(" None")).dropna()
    df=df.astype({' activity':'float64'})
    #plot the distibution plot 

    plt.figure(figsize=(16,5))
    sns.distplot(df[' activity'])
    
    plt.show()

    #box-plot

    sns.boxplot(df[' activity'])
    plt.show()
    
    
    #IQR
    percentile25 = df[' activity'].quantile(0.25)
    percentile75 = df[' activity'].quantile(0.75)
    iqr = percentile75 - percentile25

    upperlimit = percentile75 + 1.5 * iqr
    lowerlimit = percentile25 + 1.5 * iqr

    #finding outliers

    print(df[df[' activity'] > upperlimit])
    print(df[df[' activity'] < lowerlimit])
    
    #trimming
    new_df = df[df[' activity'] < upperlimit]
    print(new_df.shape)

    #compare plots after trimming
    plt.figure(figsize=(16,5))
    plt.subplot(2,2,1)
    sns.distplot(df[' activity']) 
    plt.subplot(2,2,2)  
    sns.boxplot(df[' activity'])
    plt.subplot(2,2,3)
    sns.distplot(new_df[' activity'])
    plt.subplot(2,2,4)
    sns.boxplot(new_df[' activity'])
    plt.show()

    #capping

    new_df_cap = df.copy()
    new_df_cap[' activity'] = np.where(
    new_df_cap[' activity'] > upperlimit,
    upperlimit,
    np.where(
        new_df_cap[' activity'] < lowerlimit,
        lowerlimit,
        new_df_cap[' activity']
        )
    )

    #compare after capping
    plt.figure(figsize=(16,5))
    plt.subplot(2,2,1)
    sns.distplot(df[' activity']) 
    plt.subplot(2,2,2)  
    sns.boxplot(df[' activity'])
    plt.subplot(2,2,3)
    sns.distplot(new_df_cap[' activity'])
    plt.subplot(2,2,4)
    sns.boxplot(new_df_cap[' activity'])
    plt.show()
    new_df.to_csv("y_preprocessed.csv",index=False)