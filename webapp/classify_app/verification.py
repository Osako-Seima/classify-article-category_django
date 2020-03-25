from models import NaiveBayes

import numpy as np
import pandas as pd
from dfply import *

target = NaiveBayes()

verification_data = np.load('./data/verification_data.npy' ,allow_pickle=True).tolist()
df = pd.DataFrame.from_dict(verification_data, orient='index').reset_index().rename(columns={'index':'link', 0:'true_label'})

verification_link = []
predict_caterogy = []
for link in df['link']:
  category = target.classify(link)
  verification_link.append(link)
  predict_caterogy.append(category)

predict_data = dict(zip(verification_link, predict_caterogy))
predict_data = pd.DataFrame.from_dict(predict_data, orient='index').reset_index().rename(columns={'index':'link', 0:'pre_label'})
verification = pd.merge(df, predict_data, on='link')

verification = verification >> mutate(flg = if_else(X.true_label == X.pre_label, 1, 0))
print(verification['flg'].sum()/160)