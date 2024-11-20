import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
# Define the BERT tokenizer and model
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import os
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
# Tokenize and encode the sentences
class MyDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, inference=False):
        self.data = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.inference = inference


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # year,month,day,country,title,text

        year = self.data.iloc[index]['year']
        month = self.data.iloc[index]['month']
        day = self.data.iloc[index]['day']
        country = self.data.iloc[index]['country']
        title = self.data.iloc[index]['title']
        text = self.data.iloc[index]['title']
        if self.inference == False:         
            label = self.data.iloc[index]['label']

        # Tạo input_text rõ ràng và ngữ nghĩa
        # input_text = (
        #     f"On {year}-{month:02d}-{day:02d}, in {country}, an event titled '{title}' was reported. "
        #     f"Here is the full context: {text} [SEP]"
        # )
        # input_text = f"On {year}-{month:02d}-{day:02d}, '{title}' occurred in {country}. Context: {text}"
        input_text = f"In {year}, '{title}' happened in {country}. Context: {text} [SEP]"

        # print(input_text)

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        if self.inference == False:   
            return {
                'text': input_text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else: 
            return {
                'text': input_text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()
            }