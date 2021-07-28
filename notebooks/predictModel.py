############################################################################################
################################## 07. Model Predictions #####################################
from typing import Tuple, List
from model import RNNPOSTagger
# from dataset import WSJDataset, vocabulary
from fetchDataset import myDataset
from prepareDictionary import PennTreeBankDictionary
ds = PennTreeBankDictionary()
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle,gzip

############################### 01. Look-up dictionaries ####################################
word_to_idx, idx_to_word, pos_to_idx, idx_to_pos = ds.vocabulary()

validation_dataset = DataLoader(dataset=myDataset("PennTreeBankValid"),batch_size=16,shuffle=False)

# read this seq2seq model: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html --> for understanding embedding dimension and output dimension  
# VOCAB_SIZE = len(word_to_idx)+1
# EMBED_DIM = 100
# HIDDEN_DIM = 64
# NUM_LAYERS = 1
# NUM_OF_CLASSES = len(pos_to_idx)+1
# N_EPOCHS = 5
# LEARNING_RATE = 0.005
# BATCH_SIZE = 64

VOCAB_SIZE = len(word_to_idx)+1
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_OF_CLASSES = len(pos_to_idx)
N_EPOCHS = 5
LEARNING_RATE = 0.025#0.08
BATCH_SIZE = 16

print(f"Our vocab size to the model is therefore: {VOCAB_SIZE}")
################################### 02. NN Model  ########################################

print("Step 02. builing the model...")
model = RNNPOSTagger(embedding_dimension= EMBED_DIM,
                            vocabulary_size=VOCAB_SIZE,
                            hidden_dimension=HIDDEN_DIM,
                            num_of_layers=NUM_LAYERS,
                            dropout=0.1,#0.25,
                            output_dimension=NUM_OF_CLASSES)
print("----------------------------------------------------------------")
print("Done! here is our model:")
print(model)
print("----------------------------------------------------------------")

################################## 03. load trained model ###############################
model.load_state_dict(torch.load("C:/Users/rahin/projects/WSJ-POS-tagger/notebooks/PennPOSmodel.pth"))
model.eval()

################################## 03. Predictions ###############################
print("Lets make predictions")

def token_pipeline(x):
    
    if len(x) < 50:
        for i in range(1,60-len(x)):
            x.append('PADDING')
    return [word_to_idx[tok] for tok in x]

def token_reverse_pipeline(x):
    return [idx_to_word[idx] for idx in x]

def pos_reverse_pipeline(x):
    return [idx_to_pos[idx] for idx in x]

def pos_pipeline(x):
    return [pos_to_idx[pos] for pos in x]
##############################################################################################
def predict_full_validation_dataset(example_sentence) -> Tuple[List, List]:
    sentence_to_tensor = example_sentence.unsqueeze(1).T
    with torch.no_grad():
        output = model(sentence_to_tensor)
        predicted_output = torch.argmax(output, dim=2)
        example_predicted_labels = pos_reverse_pipeline(predicted_output.tolist()[0])
        example_sentence_words = token_reverse_pipeline(sentence_to_tensor.tolist()[0])

    # return example_predicted_labels
    return example_sentence_words,example_predicted_labels
###############################################################################################
def predict_example(example_sentence, example_actual_labels):

    

    # preprocessing:-
    sentence_to_token = token_pipeline(example_sentence)
    sentence_to_tensor = torch.tensor(sentence_to_token).unsqueeze(1).T

    # predicted labels:-
    with torch.no_grad():
        output = model(sentence_to_tensor)
        predicted_output = torch.argmax(output, dim=2)
        print(predicted_output)
        #-------------
        print(pos_pipeline(example_actual_labels))
        # print(predicted_output.tolist()[0][:-1])
        print(predicted_output.tolist()[0][:len(example_actual_labels)])

        example_predicted_labels = pos_reverse_pipeline(predicted_output.tolist()[0])
        print("-"*100)
        print(f"Actual lables:- \n{example_actual_labels}")
        print(f"Predicted lables:- \n{example_predicted_labels[:len(example_actual_labels)]}")
        print("-"*100)
    # return example_predicted_labels[:len(example_actual_labels)]
###################################################################################################
example = [['This', 'time', ',', 'the', 'firms', 'were', 'ready', '.'],
            ['We', "'re", 'about', 'to', 'see', 'if', 'advertising', 'works', '.']]
example_labels = [['DT', 'NN', ',', 'DT', 'NNS', 'VBD', 'JJ', '.'],
                ['PRP', 'VBP', 'IN', 'TO', 'VB', 'IN', 'NN', 'VBZ', '.']]

predict_example(example_sentence=example[0],example_actual_labels=example_labels[0])
print("EXAMPLE 2")
predict_example(example_sentence=example[1],example_actual_labels=example_labels[1])

##################################################################################################
# print("Composing the result of first nn network to POS tag the dataset:")
# all_results = []
# for idx, (sample, label) in enumerate(validation_dataset):
#     for item in sample:
#         all_results.append(predict_full_validation_dataset(item))
# # print(all_results[:2])
# with gzip.open('C:/Users/rahin/projects/WSJ-POS-tagger/data/interim/validation_dataset_pos_tagged.pklz', 'wb') as f:
#     pickle.dump(all_results, f)
#     f.close()
# print("done!")
#########################################################################################    


# result comparison
all_actual_labels, all_predicted_labels = [],[] 

# def pos_reverse_pipeline(x):
#     return [idx_to_pos[idx] for idx in x]

for idx, (sample,label) in enumerate(validation_dataset):
  for sam in sample:
    all_predicted_labels.append(predict_full_validation_dataset(sam)[1])
  for lab in label:
    all_actual_labels.append(pos_reverse_pipeline(lab.tolist()))


def train_accuracy(preds, y):

    # print(len(preds)) #i get 10 samples
    accuracy_of_all_lines = []

    
    
    for pred,act in zip(preds,y):
        
        counter = 0
        
        for itemx,itemj in zip(pred,act):
          
          
          if itemx == itemj:
              counter = counter+1
        accuracy_of_this_line = counter/50

        accuracy_of_all_lines.append(accuracy_of_this_line)
    
    # print(accuracy_of_all_lines)
    acc = sum(accuracy_of_all_lines)/len(preds)

    return acc*100
print(f"Total Validation Accuracy of the mode: {train_accuracy(all_predicted_labels,all_actual_labels)}%")