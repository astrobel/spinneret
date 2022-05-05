import pandas as pd
import numpy as np

# s21_rot = pd.read_csv('S21_rotators.csv')

# sample_size = len(s21_rot['KIC'])

# sets = np.ones(sample_size)
# n1 = int(np.round(sample_size*0.1))
# n2 = int(np.round(sample_size*0.2))
# sets[:n1] = 2 # 10%
# sets[n1:n1+n2] = 3 # 20%
# np.random.shuffle(sets)

# train = s21_rot.iloc[np.where(sets==1)]
# test = s21_rot.iloc[np.where(sets==2)]
# validate = s21_rot.iloc[np.where(sets==3)]

# train.to_csv('S21_train.csv')
# test.to_csv('S21_test.csv')
# validate.to_csv('S21_validate.csv')

# s21_nonrot = pd.read_csv('S21_nonrotators.csv')
#
# sample_size = len(s21_nonrot['KIC'])
#
# sets = np.ones(sample_size)
# n1 = int(np.round(sample_size*0.1))
# n2 = int(np.round(sample_size*0.2))
# sets[:n1] = 2 # 10%
# sets[n1:n1+n2] = 3 # 20%
# np.random.shuffle(sets)
#
# train = s21_nonrot.iloc[np.where(sets==1)]
# test = s21_nonrot.iloc[np.where(sets==2)]
# validate = s21_nonrot.iloc[np.where(sets==3)]
#
# train.to_csv('S21n_train.csv')
# test.to_csv('S21n_test.csv')
# validate.to_csv('S21n_validate.csv')


o17_rot = pd.read_csv('O17_kelt_rotators.csv')

sample_size = len(o17_rot['TIC_ID'])

sets = np.ones(sample_size)
n1 = int(np.round(sample_size*0.1))
n2 = int(np.round(sample_size*0.2))
sets[:n1] = 2 # 10%
sets[n1:n1+n2] = 3 # 20%
np.random.shuffle(sets)

train = o17_rot.iloc[np.where(sets==1)]
test = o17_rot.iloc[np.where(sets==2)]
validate = o17_rot.iloc[np.where(sets==3)]

train.to_csv('O17_train.csv')
test.to_csv('O17_test.csv')
validate.to_csv('O17_validate.csv')
