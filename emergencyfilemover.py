import pandas as pd
import shutil, glob, sys

tess_list = glob.glob('./targetdata/*_tess.csv')

kics = [int(t.split('kic')[1].split('_')[0]) for t in tess_list]

to_move = pd.read_csv('S21n_test.csv')
kics_to_move = to_move['KIC'].values

for i, k in enumerate(kics):
    if k in kics_to_move:
        print(k)
        shutil.move(tess_list[i], f'./testsetdata/kic{k}_tess.csv')
