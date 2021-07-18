# Dictionary of Penn Tree Bank sample size

# Step 0: Import Libraries

import nltk
from nltk.corpus import treebank
import pickle

from nltk.tokenize import word_tokenize

print("creating and loading dictionaries...")
# Step 1: List of words in this sample Penn Tree Bank WSJ corpus

unique_word_list = list(set(treebank.words()))
unique_word_list.append('PADDING')

# Step 2: Create a dictionary of words and reverse look-up of words

word_to_idx, idx_to_word =  {}, {}

for idx, word in enumerate(unique_word_list):
    word_to_idx[word] = idx  # word --> idx
    idx_to_word[idx] = word  # idx --> word

sentences = treebank.tagged_sents()
pos_tags_list = []
for sentence in sentences:
    for word in sentence:
        pos_tags_list.append(word[1])

unique_pos_tags = list(set(pos_tags_list))
unique_pos_tags.append("PADDING")

pos_to_idx, idx_to_pos =  {}, {}

for idx, pos in enumerate(unique_pos_tags):
    pos_to_idx[pos] = idx
    idx_to_pos[idx] = pos

# Step 3: Export the dictionary to files

# (a) Word Look-up dictionaries:-
vocab_to_file = open("C:/Users/rahin/projects/WSJ-POS-tagger/data/raw/wsj_word_to_idx.pkl","wb")
pickle.dump(word_to_idx, vocab_to_file)
vocab_to_file.close()
vocab_to_file = open("C:/Users/rahin/projects/WSJ-POS-tagger/data/raw/wsj_idx_to_word.pkl","wb")
pickle.dump(idx_to_word, vocab_to_file)
vocab_to_file.close()

# (b) POS tags look-up dictionaries:-
pos_tags_to_file = open("C:/Users/rahin/projects/WSJ-POS-tagger/data/raw/wsj_pos_to_idx.pkl","wb")
pickle.dump(pos_to_idx, pos_tags_to_file)
pos_tags_to_file.close()
pos_tags_to_file = open("C:/Users/rahin/projects/WSJ-POS-tagger/data/raw/wsj_idx_to_pos.pkl","wb")
pickle.dump(idx_to_pos, pos_tags_to_file)
pos_tags_to_file.close()

print("done...")
####################################################################################################