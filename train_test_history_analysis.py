import pickle
import glob
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

train_thistory = []
test_shistory =[]

for i in range(12):

    print(f'\nepoch: {i}\n')
    train_history_filename_list = glob.glob(f".//model//loss_train_epch_{i}*.history")
    train_history = []
    sum = 0
    mean = []
    dloss = []
    for filename in train_history_filename_list:

        with open(filename, 'rb') as f:
            historyList = pickle.load(f)
        
        df_thistory = pd.DataFrame(historyList, columns=['total-loss', 'gen-loss', 'disc-loss'])
        for x in tqdm.tqdm(df_thistory.astype(float)['disc-loss']):
            sum = sum + x
            dloss.append(x)
            mean.append(sum/len(dloss))

        pd.DataFrame(mean).plot(kind='line')
        plt.show()

        train_thistory.append(df_thistory)

    print()

    test_history_filename_list = glob.glob(f".//model//loss_test_epch_{i}*.history")
    test_history = []
    sum = 0
    mean = []
    dloss = []

    for filename in test_history_filename_list:

        with open(filename, 'rb') as f:
            historyList = pickle.load(f)
        
        df_shistory = pd.DataFrame(historyList, columns=['total-loss', 'gen-loss', 'disc-loss'])
        for x in tqdm.tqdm(df_shistory.astype(float)['disc-loss']):
            sum = sum + x
            dloss.append(x)
            mean.append(sum/len(dloss))

        pd.DataFrame(mean).plot(kind='line')
        # df_shistory.astype(float)['disc-loss'].plot(kind='line')
        plt.show()

        test_shistory.append(df_shistory)

    print()

print('\ntrain loss')
dft = pd.concat(train_thistory)
dloss=[]
mean = []
sum = 0
for x in tqdm.tqdm(dft.astype(float)['disc-loss']):
    sum = sum + x
    dloss.append(x)
    mean.append(sum/len(dloss))
pd.DataFrame(mean).plot(kind='line')
# dft.astype(float)['disc-loss'].plot(kind='line')
plt.show()
print('\ntest loss')
dfs = pd.concat(test_shistory)
dloss=[]
mean = []
sum = 0
for x in tqdm.tqdm(dfs.astype(float)['disc-loss']):
    sum = sum + x
    dloss.append(x)
    mean.append(sum/len(dloss))
pd.DataFrame(mean).plot(kind='line')
plt.show()
    
