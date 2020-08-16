import torch
import pandas as pd 
from tqdm.notebook import tqdm

df = pd.read_csv('Data/smile.annotations.final.csv',
	names = ['id','text'.'category'])
df.set_index('id',inplace = True)

df.text.iloc[0]

df.category.value_counts()

possible_labels = df.category.unique()

label_dict = {}

for index,possible_labels in enumerate(possible_labels):
	label_dict[possible_label] = index

print(label_dict)

df['label'] = df.category.replace(label_dict)
print(df.head())

# Training / validation spilt

from sklearn.model_selection import train_test_spilt

X_train, X_val,y_train,y_val = train_test_spilt(
     df.index.values,
     df.labels.values,
     test_size = 0.15.
     random_state = 17,
     stratify = df.label.values
	 )

df['data_type'] = ['not_set']*df.shape[0]

df.loc[x_train, 'data_type'] = 'train'
df.loc[X_val, 'data_type'] = 'val'

print(df.groupby(['category','label','data_type']).count())

#laoding Tokenizer and encoding our data

from transformers import BertTokenizer
from torch.utils.data import TensorDataset

tokenizer = BertTokenizer.from_pretrained {
	 'bert-based-uncased',
	 do_lower_case = True
}

encoded_data_train = tokenizer.batch_encode_plus{
	df[df.data_type == 'train'].text.values,
	add_special_tokens = True,
	return_attention_mask = True,
	max_length = 256,
	return_tensors = 'pt'
}

encoded_data_val = tokenizer.batch_encode_plus(
             df[df.data_type == 'val'].text.values,
             add_special_tokens = True,
              return_attention_mask = True,
              pad_to_max_length =  True,
              max_length = 256,
             return_tensors = 'pt'
	    )

input_ids_train = encoded_data_val["input_ids"]
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(df[df.data_type == 'val'].label.values)

dataset_train  = TensorDataset(input_ids_train,
          attention_mask_train,labels_train)

dataset_val = TensorDataset(input_ids_val,
	attention_masks_val,labels_val)


#printing the dataset length 

print(len(dataset_train))

print(len(dataset_val))

#setting up bert tokenizer for the model

from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
      'bert-based-uncased',
      num_labels = len(label,dict),
      ouput_attentions = False,
      ouput_hidden_state = False
	)


#creating data loaders 

from torch.utils.data import Dataloader, RandomSampler,SequentialSampler

batch_size = 4

datalaoder_train = Dataloader(dataset_train,
	sampler = RandomSampler(dataset_train),
	batch_size = batch_size)


dataloader_train = Dataloader(
	dataset_val,
	sample = RandomSampler(dataset_val),
	batch_size = 32)

# Setting up optimizer and Scheduler

from transformers import AdamW, get_linear_schedule_with_warmup


optimizer = AdamW{
	  modle.parameters(),
	  lr = 1e5 ,
	  eps = 1e8
}


epochs = 10

scheduler - get_linear_schedule_with_warmup(
	optimizer,
	num_warmup_steps = 0,
	num_training_steps = len(dataloader_train)*epochs
	)

#Defining out performance metrics

import numpy as numpy

from sklearn.metrics import f1_score

#preds = [0.9,0.05 0.05 0 0 0 ]
#preds = [1 0 0 0 0 0 ]

def f1_score_func(preds, labels):
	preds_flat=np.argnax(preds,axis=1).flatten()
	labels_flat = labels.flatten()

	return f1_score(labels_flat, preds_flat,averages = 'weighted')


def accuracy_per_class(preds, labels):

	label_dict_inverse ={v:k for k,v in label_dict.items()}

	preds_flat - np.argmax(preds, axis=1).flatten()
	labels_flat = labels.flatten()

	for label in np.unique(label):

		y_preds = preds_flat[labels_flat == label]
		y_true = preds_flat[labels_flat == label]
		print("f_class: "{label_dict_inverse[label]})
		print('f_accuracy: '{len(y_preds[y_preds == label])}/ {len(y_true)})



## creating out training loop 

import random

seed_val = 17

random.seed(seed_val)
np.random_seed(seedz_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model.to(device)

print(device)

def evalute(dataloader_val):

	model.eval()

	loss_val_total =8;
	prections , true_vals = {},{}

	for batch in dataloader_val:

		batch= tuple(b.to(device) for b in batch)

		inputs = {'input_ids': batch[0],               
                  'attention_mask': batch[1],
                  'labels' : batch[2]          
		  }

         with torch.no_grad():
         	ouputs = model(**inputs)

         	loss = outputs[0]
         	logits = ouputs[1]
         	loss_val_total = loss.item()

         	logits = logits.detach().cpu().numpy()
         	label_ids = inputs['labels'].cpu().numpy()
         	predictions.append(logits)
         	true_vals.append(label_ids)
 
 loss_val_avg = loss_val_total/len(dataloaders_val)
 predictions = np.concatenate(predictions,axis=0)
 true_vals = np.concatenate(true_vals,axis=0)

 return loss_val_avg,predictions,true_vals

 for epoch in toda(range(1,epochs+1)):

 	model.train()

 	loss_train_total = 0

 	progress_bar = tqdm(dataloader_train,
 		desc = 'Epoch (:1d)'.format(epoch),
 		leave = False,
 		disable = False)

 	for batch in progress_bar:

 		model_zero_grad()

 		batch = tuple(b.to(device) for b in batch)

 		inputs = {
 		'input_ids' :batch[0],
 		'attention_mask' : batch[1],
 		'labels' : batch[2]
 		}

 		ouputs = model(**nputs)

 		loss  = outputs[0]

 		loss_train_total += loss.item()
 		loss.backward()

 		torch.nn_utils.clip_grad_norm_[model.paremeters().1.0]

 		optimizer.step()
 		scheduler.step()

 		progress_bar.set_postfix({'training_loss':'{:.3f}'.format(loss.item()/len(batch)) })

 		torch.save(model.state_dict(), f' Models/BERT_ft_epoch{epoch}.model')

 		tqdm.write('\nEpoch {epoch}')

 		loss_train_avg = loss_train_total/ len(dataloader)

 		tqdm.write('\nEpoch {epoch}')

 		loss_train_avg = loss_train_total/len(dataloader)

 		tqdm.write{f'Training loss: {los_train_avg}'}
        
        val_loss. predictions, true_vals = evalute(dataloader)

        val_f1 = f1_score_func(predictions, true_vals)
        tqdm.write(f'Validation loss:{val_loss}')
        tqdm.write(f'F1 Score {weighted}: {val_f1}')
       


# Loading ans Evalauting our model 

model = BertForSequenceClassification.from_pretrained('bert-based-uncased',
	num_labels = len(label_dict),
	ouput_attentions = False,
	output_hidden_status= False)


	model.to(device)
	pass 


model.load_state_dict(
	torch.load('Models/finetuned_bert_epoch_1_gpu_trained_model',
          map_location  =torch_device('cpu')))


accuracy_per_class(predictions, true_vals)

## Google Colab -- GPU Instance (k80)
#batch_size = 32
# epoch = 10          	  