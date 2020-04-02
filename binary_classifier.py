import pandas as pd
import prettytable as pt
import numpy as np 
from tabulate import tabulate
table = pd.read_table('HDTB_pre_release_version-0.05/IntraChunk/CoNLL/utf/conversation/file-andhe_ki_lathi-1507111038.dat', skiprows=0, names=['ID', 'FORM', 'LEMMA', 'CPOSTAG', 'POSTAG', 'FEATS', 'HEAD', 'DEPREL', 'PHEAD', 'PDEPREL'])


# print(tabulate(table, headers=['ID','FORM','LEMMA','CPOSTAG','POSTAG','FEATS','HEAD','DEPREL','PHEAD','PDEPREL'], showindex=False))
# print(table["FORM"])
feature_array = table[['FEATS']].to_numpy()
np.savetxt('features.txt', feature_array, fmt="%s", delimiter='|', newline='\n', header='', footer='', comments='# ', encoding=None)

# print(feature_array)
features = pd.read_csv('features.txt', sep='|', skiprows=0)

final = pd.concat([table, features], axis = 1)

print(tabulate(final,  showindex=False))
