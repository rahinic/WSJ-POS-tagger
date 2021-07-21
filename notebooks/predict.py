############################################################################################
################################## 07. Model Predictions #####################################
from model import RNNPOSTagger
from dataset import WSJDataset, vocabulary
import torch
from torch.utils.data import DataLoader
import numpy as np
from nltk.corpus import treebank

############################### 01. Look-up dictionaries ####################################
vocab = vocabulary()
word_to_idx, pos_to_idx = vocab.load_dictionaries()
idx_to_word, idx_to_pos = vocab.load_reverse_dictionaries()

# read this seq2seq model: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html --> for understanding embedding dimension and output dimension  
VOCAB_SIZE = len(word_to_idx)+1
EMBED_DIM = 256
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_OF_CLASSES = len(pos_to_idx)
N_EPOCHS = 30
LEARNING_RATE = 0.0001
BATCH_SIZE = 256

print(f"Our vocab size to the model is therefore: {VOCAB_SIZE}")
################################### 02. NN Model  ########################################

print("Step 02. builing the model...")
model = RNNPOSTagger(embedding_dimension= EMBED_DIM,
                            vocabulary_size=VOCAB_SIZE,
                            hidden_dimension=HIDDEN_DIM,
                            num_of_layers=NUM_LAYERS,
                            dropout=0.15,
                            output_dimension=NUM_OF_CLASSES)
print("----------------------------------------------------------------")
print("Done! here is our model:")
print(model)
print("----------------------------------------------------------------")

################################## 03. load trained model ###############################
model.load_state_dict(torch.load("C:/Users/rahin/projects/WSJ-POS-tagger/notebooks/WSJPOSmodel.pth"))
model.eval()

print("Lets make predictions")

sentences = treebank.tagged_sents()
validation_dataset = DataLoader(dataset=WSJDataset(sentences[3901:]), batch_size=256, shuffle=True)

def predict(sentence, model):

    # token idx to tensor conversion
    idx_to_torch01 = sentence

    with torch.no_grad():
        output = model(idx_to_torch01)
        # print(output.size())
        predicted_ouput=torch.argmax(output,dim=2)
        # print(predicted_ouput)
        predicted_labels = []

        for pred in predicted_ouput:
            for i in pred:
                predicted_labels.append(idx_to_pos[int(i)])

        return output, predicted_labels

model = model.to("cpu")
# ==============================================================================
# ACCURACY & PRECISION CALCULATIONS

total_accuracy = []
length_of_sentence = []
print("="*100)
print("Accuracy calculations:")
def model_accuracy_precision():

    """ returns accuracy of the model using the validation dataset."""

    for idx, sample in enumerate(validation_dataset):

        for ex in sample: # in each sample
            x, y = ex[0], ex[1].tolist()[0] # take sentence tensors as it is and convert the labels to list
            actual_labels = []

            for idx in y:
                actual_labels.append(idx_to_pos[int(idx)]) # convert back the label idx list to actual POS tables. 

            probsy, predictedy = predict(x, model) #predicted labels
            correct = np.array(actual_labels) == np.array(predictedy) # boolean comparison

            total_accuracy.append(correct.sum())
            length_of_sentence.append(len(x))

            acc=sum(total_accuracy)/sum(length_of_sentence)*100

    return round(acc,2)

print(f"Total Accuracy of our model is: {model_accuracy_precision()}%")
# ======================================================================================

# extract one example:
for idx, sample in enumerate(validation_dataset):

    if idx > 3:
        break
    sentence, label = sample[5][0].tolist()[0], sample[5][1].tolist()[0]

# from this example, let us construct the words back again:

actual_labels, actual_sentence = [], []

for idxw, idxp in zip(sentence, label):
    actual_labels.append(idx_to_pos[idxp])
    actual_sentence.append(idx_to_word[idxw])

print("="*100)
print("Example Sentence and its POS tags:\n" + f"{actual_sentence} \n {actual_labels}")
print("="*100)
print("After token to index conversion of the labels:")
print(f"{label}")
example = sample[5][0]
probsy, predictions = predict(example, model)
probsy_np = probsy.cpu().detach().numpy()
probsy_np =  np.squeeze(probsy_np, axis=0)
print("="*100)
print(predictions)

predicted_labels_to_idx = []
for item in predictions:
    predicted_labels_to_idx.append(pos_to_idx[item])

print(predicted_labels_to_idx)

print(idx_to_pos)
# print(probsy_np)

# for item in probsy_np:
#     print(item)

# # ====================================================================================
# # Export the results of our predictions and their corresponding probabilities.
# # This will be used as input to the viterbi algorithm

# # Step 1: Export our BIOES predictions
# FILEPATH = "data/processed"

# textfile = open(FILEPATH+"/sentence.txt", "w")
# for element in predictions:
#     textfile.write(element + "\n")
# textfile.close()

# # Step 2: Export the individual probability of each BIOES tag, given each words+POS tags predictions

# np.save(FILEPATH+"/tags_probabilities01.npy", probsy_np)