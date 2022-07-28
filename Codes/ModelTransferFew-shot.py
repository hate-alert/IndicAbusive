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
import os
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


# In[23]:


#get_ipython().system('nvidia-smi')


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
        batch = tuple(t.to(device, dtype = torch.long) for t in batch)
  
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



globalpath="./"



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


alltrainDatasetPaths = {
'Bn':globalpath+"Bengali/bn_train.csv",
'Hi':globalpath+"Hindi/hi_train.csv",
'Hn-En': globalpath+"Hindi/Code-Mixed/hiEn_train.csv",
'Ka-En': globalpath+"Kannada/Code-Mixed/kaEn_train.csv",
'Ma-En': globalpath+"Mala/Code-Mixed/maEn_train.csv",
'Mr': globalpath+"Marathi/mr_train.csv",
'Tm': globalpath+"Tamil/tm_train.csv",
'Tm-En': globalpath+"Tamil/Code-Mixed/tmEn_train.csv",
'Ur-En': globalpath+"Urdu/Code-Mixed/urEn_train.csv",
'Ur': globalpath+"Urdu/ur_train.csv",
'En': globalpath+"English/en_train.csv"
}


allvalDatasetPaths = {
'Bn':globalpath+"Bengali/bn_val.csv",
'Hi':globalpath+"Hindi/hi_val.csv",
'Hn-En': globalpath+"Hindi/Code-Mixed/hiEn_val.csv",
'Ka-En': globalpath+"Kannada/Code-Mixed/kaEn_val.csv",
'Ma-En': globalpath+"Mala/Code-Mixed/maEn_val.csv",
'Mr': globalpath+"Marathi/mr_val.csv",
'Tm': globalpath+"Tamil/tm_val.csv",
'Tm-En': globalpath+"Tamil/Code-Mixed/tmEn_val.csv",
'Ur-En': globalpath+"Urdu/Code-Mixed/urEn_val.csv",
'Ur': globalpath+"Urdu/ur_val.csv",
'En': globalpath+"English/en_val.csv"
}


alltestDatasetPaths = {
'Bn':globalpath+"Bengali/bn_test.csv",
'Hi':globalpath+"Hindi/hi_test.csv",
'Hn-En': globalpath+"Hindi/Code-Mixed/hiEn_test.csv",
'Ka-En': globalpath+"Kannada/Code-Mixed/kaEn_test.csv",
'Ma-En': globalpath+"Mala/Code-Mixed/maEn_test.csv",
'Mr': globalpath+"Marathi/mr_test.csv",
'Tm': globalpath+"Tamil/tm_test.csv",
'Tm-En': globalpath+"Tamil/Code-Mixed/tmEn_test.csv",
'Ur-En': globalpath+"Urdu/Code-Mixed/urEn_test.csv",
'Ur': globalpath+"Urdu/ur_test.csv",
'En': globalpath+"English/en_test.csv"
}



allLangs = ['Bn','Hi', 'Hn-En', 'Ka-En', 'Ma-En', 'Mr', 'Tm', 'Tm-En', 'Ur-En', 'Ur', 'En']
allLangsBack = ['Bn','Hi', 'Hn-En', 'Ka-En', 'Ma-En', 'Mr', 'Tm', 'Tm-En', 'Ur-En', 'Ur', 'En']


# In[20]:


import numpy as np
def getProb(temp):
    t = np.exp(temp)
    return t[1]/(sum(t))


# In[28]:


allPredModelLang ={}


# In[ ]:


def trainModel(model):
    model.cuda()
    bestValAcc  = 0
    bestValMF1  = 0
    besttest_df  = None
    bestEpochs = -1

    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    epochs = veParams['epochs']
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []

    for epoch_i in range(0, epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple.
            loss = outputs[0]

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        origValValue = []
        preValValue  = []
        for batch in validation_dataloader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            origValValue.extend(label_ids)
            valPred_flat = np.argmax(logits, axis=1).flatten()
            preValValue.extend(valPred_flat)
            # Track the number of batches
            nb_eval_steps += 1

        # Report the final accuracy for this validation run.
        valMf1Score = f1_score(origValValue, preValValue, average='macro')
    #    if ((eval_accuracy/nb_eval_steps) > bestValAcc):
        if (valMf1Score > bestValMF1):
            bestEpochs = epoch_i
            bestValMF1 = valMf1Score
            bestValAcc  = eval_accuracy/nb_eval_steps
            besttest_df = getPerformanceOfLoader(model, test_dataloader)
        print("  Accuracy: {0:.2f}, Macro F1 Score: {0:.2f} ".format(eval_accuracy/nb_eval_steps,valMf1Score))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))

    print("")
    print("Training complete!")
    print("Best Epoch,", bestEpochs)
    return besttest_df


# In[17]:


def getTrainDatatoTrain(origTrain, count, random_stateIn=2021):
    hate = origTrain[origTrain['label']==1]
    nonhate = origTrain[origTrain['label']==0]
    hateSame = hate.sample(n = count, random_state=random_stateIn)
    nonHateSame = nonhate.sample(n = count, random_state=random_stateIn)
    df = pd.concat([hateSame, nonHateSame], ignore_index=True)
    dfSamp = df.sample(frac = 1, random_state=random_stateIn)
    return dfSamp


# In[ ]:

import os

for mod in allLangs:
    print("Model", mod)
    for language in allLangsBack:
        print("language",language)
        if mod == language:
            print("SameLanguage")
            continue
        origTrainData = pd.read_csv(alltrainDatasetPaths[language])
        valData = pd.read_csv(allvalDatasetPaths[language])
        testData = pd.read_csv(alltestDatasetPaths[language])
        
        origTrainData.dropna(subset = ["text"], inplace=True)
        valData.dropna(subset = ["text"], inplace=True)
        testData.dropna(subset = ["text"], inplace=True)

        X_v, y_v = list(valData['text']), list(valData['label'])
        X_ts, y_ts = list(testData['text']), list(testData['label'])
        tokenizer = BertTokenizer.from_pretrained(allModelPaths[mod], do_lower_case = True)
        
        X_val, attention_masks_val, y_val = getFeaturesandLabel(X_v,y_v)
        X_test, attention_masks_test, y_test = getFeaturesandLabel(X_ts,y_ts)
        
        val_data = TensorDataset(X_val, attention_masks_val, y_val)
        test_data = TensorDataset(X_test, attention_masks_test, y_test)
        
        val_sampler = SequentialSampler(val_data)
        test_sampler = SequentialSampler(test_data)
        
        validation_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = veParams['batch_size'])
        test_dataloader = DataLoader(test_data, sampler = test_sampler, batch_size = veParams['batch_size'])
        
        
        for i in range(0,3):
            print("mod", mod, "language", language, i)
            if (os.path.isfile("Model_"+mod+"_"+language+"_"+str(i)+".csv") == True):
                print("Done")
                continue
            trainData = getTrainDatatoTrain(origTrainData, 64, 2021+i)
            X_tn, y_tn = list(trainData['text']), list(trainData['label'])
            X_train, attention_masks_train, y_train = getFeaturesandLabel(X_tn,y_tn)
            train_data = TensorDataset(X_train, attention_masks_train, y_train)
            train_sampler = RandomSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = veParams['batch_size'])
            
            model = select_model(veParams['what_bert'], allModelPaths[mod], [0.5, 0.5])
            besttest_df = trainModel(model)
            testData['pred']=besttest_df['target']
            testData.to_csv("Model_"+mod+"_"+language+"_"+str(i)+".csv")

