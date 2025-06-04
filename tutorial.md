"""
this is how i dowload the dataset into the jupiter lab after is uploaded into the s3 bucket
for sagemaker
"""


import pandas as pd


s3_path = 's3://hugging-face-multiclass-textclassification-bucket/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])
df.head() # display the first few rows of the dataframe
## it can use df.describe or df.info() to get more information about the dataset

Alawasy make a copy of the data set when working on it becuase i might lead to data loss 

df.work= df.copy ()
df.work = df,work[[ 'Title']]. List the columns you will need only to train the models, if if you only need even lees from what u intialyy declare

you can use dictionaries to convert the name of collumns into a more describing name. 
Ex: 
my dict = {

    'e': Entretaiment 
    's': sports

}

Then you will have to apply this to the dataset

ex: 
def updatecategory (x):
    return my_dict[x]
df.work['Category'] = df.work['Category'].apply(lambda x: update_category (x))


import random

def get_random_title_by_category(category):
    filtered_df = df_work[df_work['CATEGORY'] == category]
    return filtered_df['TITLE'].sample().values[0]

category = 'Entertainment'
random_title = get_random_title_by_category(category)

print(random_title)


From all datsets we should create a bar chart to see the distribution and make sure the data is equally distributed

import seaborn as sns
import matplotlib.pyplot as plt

# bar chart
plt.figure(figsize=(10, 6))
sns.countplot(data=df_work, x='CATEGORY', order=df_work['CATEGORY'].value_counts().index)
plt.title('Distribution of Categories')
plt.xticks(rotation=45)
plt.show()

After this depending on how balnace the data is we could apply techniques to balance the data 

category_counts = df_work['CATEGORY'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Each Category')
plt.show()




Training process
you most create a S3 folder for the new model that the sagemaker will create 
create the folder inside the same folder as the data was created 
copy the Se URI to put it in the inside the hugging face stimatior
! pip install transformers 
import torch
import transformers
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker import get_execution_role

role = get_execution_role()


from sagemaker.huggingface import HuggingFace

huggingface_estimator = HuggingFace(
    entry_point='script.py',  # this will have the information about the model and training
    source_dir = './',
        role = role,
        instance_count = 1,
        instance_type = 'ml.p2.xlarge'-- > change this as it has gpu
        transformers_version = '4.6',
        pytorch_version = '1.8',
        output_path = 's3://hugging-face-multiclass-textclassification-bucket/output/',
        py_version = 'py36',
        hyperparameters = {
        'epochs': 2,
        'train_batch_size': 4,
        'valid_batch_size': 2,
        'learning_rate': 1e-05
},
        enable_sagemaker_metrics= True)

huggingface_stimator.fit()


script function to call on huggingface_stimator
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import argparse
import os
import pandas as pd

s3_path = 's3://hugging-face-multiclass-textclassification-bucket/training_data/newsCorpora.csv'
df = pd.read_csv(s3_path, sep='\t', names=['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP'])

df = df[['TITLE' 'CATEGORY']]
my dict = {

    'e': Entretaiment 
    's': sports

}

def update_cat(x):
    return my_dicc[x]

df.work['Category'] = df.work['Category'].apply(lambda x: update_cat (x))    
print(df)


train a model in a small datasate to test that it works so you do not loos all the time and money for training 

# This is just a tip
df = df.sample(frac=0.05, random_state=1)

df = df.reset_index(drop=True)
# This is where the tip ends

# Encoding Categorical labes to numeric values
for this speficic case he encoded the data to class numbers for the model to learn better and fast. Like entretaimenet = 0 , health = 1, etc

encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x] = len(encode_dict)
    return encode_dict[x]

df['ENCODE_CAT'] = df['CATEGORY'].apply(lambda x: encode_cat(x))

the test also needs to be tokenized 
