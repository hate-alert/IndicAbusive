#!/usr/bin/env python
# coding: utf-8

# In[23]:


import torch
import torch.nn as nn
import json
import random
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import time
import datetime
import random
import re
import numpy as np
import emoji
from transformers import *
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from sklearn.metrics import accuracy_score,f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from sklearn.metrics import *
from sklearn.utils import *
import pandas as pd
# If there's a GPU available...
if torch.cuda.is_available():    
    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


def Diff(a,b):
    return list(set(a) -set(b))

def re_sub(pattern, repl,text):
    return re.sub(pattern, repl, text)


def preprocess_sent(sent):
    sent = re.sub(r"http\S+", " ", sent.lower())
    sent = re.sub(r"@\S+", "@user", sent)

    #print(sent)
    sent = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "",sent)
    sent = emoji.demojize(sent)
    sent = re_sub(r"[:\*]", " ",sent)
    sent = re.sub(r"[<\*>]", " ",sent)
    sent = sent.replace("&amp;", " ")
    sent = sent.replace("ðŸ¤§", " ")
    sent = sent.replace("\n", " ")
    sent = sent.replace("ðŸ˜¡", " ")
    return sent

def evalMetric(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    mf1Score = f1_score(y_true, y_pred, average='macro')
    f1Score  = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    area_under_c = auc(fpr, tpr)
    recallScore = recall_score(y_true, y_pred)
    precisionScore = precision_score(y_true, y_pred)
    return {"accuracy": accuracy, 'mF1Score': mf1Score, 'f1Score': f1Score, 'auc': area_under_c, 
            'precision': precisionScore, 'recall': recallScore}

#Creating an attention mask - For actual tokens its set to 1, for padding tokens its set to 0
def create_attention_masks(input_ids):
    attention_masks = []
    for seq in tqdm(input_ids):
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return np.array(attention_masks)

def getFeaturesandLabel(X,y):
    sentences = list(X)
    sentences = ["[CLS] "+ preprocess_sent(i) + " [SEP]"for i in sentences]
    tokenizer_texts = list(map(lambda t: tokenizer.tokenize(t)[:512], tqdm(sentences)))
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tqdm(tokenizer_texts)]
    input_ids = pad_sequences(sequences = input_ids, maxlen = veParams['max_length'], dtype = 'long', padding='post', truncating='post')
    attention_masks_data = create_attention_masks(input_ids)
    X_data = torch.tensor(input_ids)
    attention_masks_data = torch.tensor(attention_masks_data)
    y_data = torch.tensor(y)
    return X_data, attention_masks_data, y_data


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))
  
def fix_the_random(seed_val = 42):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed_val) 
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# In[4]:


# Class for weighted bert for sentence classification
class SC_weighted_BERT(BertPreTrainedModel):
    def __init__(self, config,weights):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.weights=weights
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights))
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


# In[14]:


def select_model(type_of_model,path,weights=None,label_list=None):
    if(type_of_model=='weighted'):
        model = SC_weighted_BERT.from_pretrained(
        path, # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification             # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
        weights=weights
    )
    else:
        print("Error in model name!!!!")
    return model


# In[16]:


def getPerformanceOfLoader(model, test_dataloader):
    model.cuda()
    model.eval()
    # Tracking variables 
    predictions , true_labels = [], []
    # Predict 
    for batch in test_dataloader:
    #print(batch)
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
  
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
  
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # Store predictions and true labels
        pred_flat = np.argmax(logits, axis=1).flatten()
        predictions.extend(pred_flat)
        true_labels.extend(label_ids)

    df = pd.DataFrame()
    df['id']=testData['text']
    df['target'] = predictions
    return df


# In[6]:


veParams={
    'max_length':128,
    'what_bert':'weighted',
    'batch_size':16,
    'is_train':True,
    'learning_rate':2e-5,
    'epsilon':1e-8,
    'random_seed':30,
    'epochs':10,
    'to_save':True,
    'frac':0.8,
}


# In[8]:


globalpath = "./IndicLang/"


# In[11]:


allModelPaths = {
'Bn':globalpath+"Bengali",
'Hi':globalpath+"Hindi",
'Hn-En': globalpath+"Hindi/Code-Mixed",
'Ka-En': globalpath+"Kannada/Code-Mixed",
'Ma-En': globalpath+"Mala/Code-Mixed",
'Mr': globalpath+"Marathi",
'Tm': globalpath+"Tamil",
'Tm-En': globalpath+"Tamil/Code-Mixed",
'Ur-En': globalpath+"Urdu/Code-Mixed",
'Ur': globalpath+"Urdu",
'En': globalpath+"English"
}


# In[12]:


allDatasetPaths = {
'Bn':globalpath+"Bengali/bn_test.csv",
'Hi':globalpath+"Hindi/hi_test.csv",
'Hn-En': globalpath+"Hindi/Code-Mixed/hiEn_test.csv",
'Ka-En': globalpath+"Kannada/Code-Mixed/kaEn_test.csv",
'Ma-En': globalpath+"Mala/Code-Mixed/maEn_test.csv",
'Mr': globalpath+"Marathi/mr_test.csv",
'Tm-En': globalpath+"Tamil/Code-Mixed/tmEn_test.csv",
'Ur-En': globalpath+"Urdu/Code-Mixed/urEn_test.csv",
'Ur': globalpath+"Urdu/ur_test.csv",
'En': globalpath+"English/en_test.csv"
}


# In[19]:


allLangs = ['Bn','Hi', 'Hn-En', 'Ka-En', 'Ma-En', 'Mr', 'Tm-En', 'Ur-En', 'Ur', 'En']
allLangsBack = ['Bn','Hi', 'Hn-En', 'Ka-En', 'Ma-En', 'Mr','Tm-En', 'Ur-En', 'Ur', 'En']


# In[20]:


import numpy as np
def getProb(temp):
    t = np.exp(temp)
    return t[1]/(sum(t))


# In[28]:


#allPredModelLang ={}


# In[30]:


for mod in allLangs:
    print("Model", mod)
    if mod not in allPredModelLang:
        allPredModelLang[mod] = {}
    tokenizer = BertTokenizer.from_pretrained(allModelPaths[mod], do_lower_case = True)
    model = select_model(veParams['what_bert'], allModelPaths[mod], [1, 1])
    for language in allLangsBack:
        if language in allPredModelLang[mod]:
            print("already done")
            continue
        print("Language", language)
        testData =  pd.read_csv(allDatasetPaths[language])
        testData.dropna(subset = ["text"], inplace=True)
        X_ts, y_ts = list(testData['text']), list(testData['label'])
        X_test, attention_masks_test, y_test = getFeaturesandLabel(X_ts,y_ts)
        test_data = TensorDataset(X_test, attention_masks_test, y_test)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = veParams['batch_size'])
        besttest_df = getPerformanceOfLoader(model, test_dataloader)
        testData['pred']=besttest_df['target']
        allPredModelLang[mod][language]=testData


# In[32]:


import pickle
with open("allPredModelLangZeroShot_csv.p", "wb") as fp:
    pickle.dump(allPredModelLang,fp)


# In[ ]:


evalMetric


# In[34]:


allPredModelLang['Bn']['Bn']


# In[39]:


for mod in allLangs:
    print("Model", mod)
    for language in allLangsBack:
        temp = evalMetric(allPredModelLang[mod][language]['label'], allPredModelLang[mod][language]['pred'])
        print("Lanuage", language, temp['mF1Score'])
    print("---------------------")


# In[ ]:




