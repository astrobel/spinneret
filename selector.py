import pandas as pd
import numpy as np
import lightkurve as lk

file_in = pd.read_csv('M14rotators.tsv', delimiter=';', skiprows=57)

# q9_indices = np.zeros(len(file_in['KIC'][2:102]), dtype='bool')
# for i, k in enumerate(file_in['KIC'][2:102]):
#     s = lk.search_lightcurve(f'KIC {k}')
#     if np.where(s.table['mission']=='Kepler Quarter 09')[0].size != 0:
#         q9_indices[i] = True
#     print(f'{k} done')

# print(q9_indices)

last = 202

d = {'KIC':file_in['KIC'][2:last], 'Teff':file_in['Teff'][2:last], 'Prot':file_in['Prot'][2:last]}
file_out = pd.DataFrame(data=d)
file_out.to_csv('M14rotators.csv')

file_in2 = pd.read_csv('M14nonrotators.tsv', delimiter=';', skiprows=53)

d = {'KIC':file_in2['KIC'][2:last], 'Teff':file_in2['Teff'][2:last], 'Prot':file_in2['Prot'][2:last]}
file_out2 = pd.DataFrame(data=d)
file_out2.to_csv('M14nonrotators.csv')