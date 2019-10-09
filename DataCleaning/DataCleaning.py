import pandas as pd
import numpy as np
import re

df = pd.read_csv('./Scraps/indeed_scrap_19-10-07-12-43.csv')
print(df.head())

def clean_job_id():
    ids = df['id']
    regex = r"(?<=vjk=)(.*$)"
    for i in ids:
        temp = re.search(regex, i)
        i = temp.group()
        i = i[:16]

    return ids

df1 = df[df['description'].isna()]
#print(df['id'])

df_filtered = df[(df['salaire'].isnull()) & (df['description'].str.contains("€")==True)] # description contient € et salaire null
df_filtered
df_filtered.to_csv('./descriptions_sans_salaire.csv')