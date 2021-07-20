import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WSJDataset, vocabulary
from model import RNNPOSTagger
from nltk.corpus import treebank
import time
import os
################################### 01. Train/Test dataset  ########################################
print("="*100)
print("01. Preparing train/test datasets:")

sentences = treebank.tagged_sents()
train_dataset = DataLoader(dataset=WSJDataset(sentences[:2700]), batch_size=16, shuffle=True)
test_dataset = DataLoader(dataset=WSJDataset(sentences[2701:3700]), batch_size=16, shuffle=True)

print("datasets ready!")
print("="*100)

################################# 02.Model Parameters ####################################
print("02. Loading Model Parameters:")
vocab = vocabulary()
word_to_idx, pos_to_idx = vocab.load_dictionaries()

# read this seq2seq model: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html --> for understanding embedding dimension and output dimension  
VOCAB_SIZE = len(word_to_idx)+1
EMBED_DIM = 100
HIDDEN_DIM = 64
NUM_LAYERS = 2
NUM_OF_CLASSES = len(pos_to_idx)
N_EPOCHS = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 16

print(f"Size of vocabulary: {VOCAB_SIZE}" + f"\tNumber of classes: {NUM_OF_CLASSES}")
##################################### 03. NN Model  ########################################

print("Step 02. builing the model...")
model = RNNPOSTagger(embedding_dimension= EMBED_DIM,
                    vocabulary_size=VOCAB_SIZE,
                    hidden_dimension=HIDDEN_DIM,
                    num_of_layers=NUM_LAYERS,
                    dropout=0.2,
                    output_dimension=NUM_OF_CLASSES)

print("Done! here is our model:")
print(model)
print("="*100)

############################# 04. Optimizer and Loss  ####################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

#define metric
def binary_accuracy(preds, y):
    
    predsx = preds.permute(0,2,1) #reshape
    predsx2 = torch.argmax(predsx, dim=2) #find POS index with max value for each token
    correct = (predsx2 == y)
    acc = correct.sum() / len(preds)
    # print(type(acc))

    return acc
    
#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)

############################## 05. NN Model Train Definition #############################

def train(model, dataset, optimizer, criterion):
    #log_interval = 500
    start_time = time.time()
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0
    epoch_dataset_length_total = 0
    epoch_dataset_length.append(len(dataset))

    model.train()

    for idx, sample in enumerate(dataset):
       
       current_samples = sample[idx][0]
       current_labels = sample[idx][1]

       optimizer.zero_grad()

       predicted_labels = model(current_samples).permute(0,2,1)
       
    #    print(f"Predicted label dimension: {predicted_labels.size()} and actual label dimension: {current_labels.size()}")
    #    print(predicted_labels)
    #    print(current_labels)
    #    predicted_labels = predicted_labels.to(torch.float)
    #    current_labels = current_labels.to(torch.float)
       loss = criterion(predicted_labels, current_labels)
       accuracy = binary_accuracy(predicted_labels, current_labels)

       loss.backward()
       optimizer.step()

       epoch_loss += loss.item()
       epoch_accuracy += accuracy.item()
    #    epoch_dataset_length_total += epoch_dataset_length.item()

    #    epoch_accuracy =0

    return epoch_loss/len(dataset), epoch_accuracy/sum(epoch_dataset_length)

##########################################################################################
################################ 06. NN Model Eval Definition ############################
def evaluate(model, dataset, criterion):
    
    start_time = time.time()
    print(start_time)

    epoch_loss = 0
    epoch_accuracy = 0
    model.eval()

    with torch.no_grad():

        for idx, sample in enumerate(dataset):
            current_samples = sample[idx][0]
            current_labels = sample[idx][1]

            predicted_labels = model(current_samples).permute(0,2,1)
            # predicted_labels = predicted_labels.to(torch.float)
            # current_labels = current_labels.to(torch.float)

            loss = criterion(predicted_labels, current_labels)
            accuracy = binary_accuracy(predicted_labels, current_labels)
            #epoch_accuracy = 0

            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss/len(dataset), epoch_accuracy/len(dataset)

##########################################################################################

################################## 06. NN Model training #####################################
#N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    print(f"Epoch #: {epoch}")
    epoch_dataset_length = []
    #train the model
    train_loss, train_acc = train(model, train_dataset, optimizer, criterion)
    
    #evaluate the model
    valid_loss, valid_acc = evaluate(model, test_dataset, criterion)
    
    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    print("-------------------------------------------------------------------")
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
    print("-------------------------------------------------------------------")

modelpath = "C:/Users/rahin/projects/WSJ-POS-tagger/notebooks"
torch.save(model.state_dict(), os.path.join(modelpath, "WSJPOSmodel.pth"))

