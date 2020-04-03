import pandas as pd
import prettytable as pt
import numpy as np 
from tabulate import tabulate
import os

files = [os.path.join(path, name) for path, subdirs, files in os.walk("dataset") for name in files]

write_file = open("data.csv", "a")

for f in files:
    table = pd.read_table(f, skiprows=0, names=['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'])


    # print(tabulate(table, headers=['ID','FORM','LEMMA','CPOSTAG','POSTAG','FEATS','HEAD','DEPREL','PHEAD','PDEPREL'], showindex=False))
    # print(table["FORM"])
    feature_array = table[['FEATS']].to_numpy()
    np.savetxt('features.txt', feature_array, fmt="%s", delimiter='|', newline='\n', header='', footer='', comments='# ', encoding=None)

    # print(feature_array)
    features = pd.read_csv('features.txt', sep='|', skiprows=0, names=['cat', 'gen', 'num', 'pers', 'case', 'vib', 'tam', 'chunkId', 'chunkType' , 'stype' , 'voicetype'])

    final = pd.concat([table, features], axis = 1)
    write_in = np.concatenate([table[['ID', 'FORM', 'LEMMA', 'HEAD', 'DEPREL']].to_numpy(), features.to_numpy()], axis=1)
    print((write_in))
    np.savetxt(write_file, write_in, fmt='%s')
    # print(tabulate(final,  showindex=False))

