import pickle
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import treebank
class WSJDataset(Dataset):

    def file_handling(self, filename: str):

        filepath = "C:/Users/rahin/projects/WSJ-POS-tagger/data/raw/"+filename
        file = open(filepath, "rb")
        content = pickle.load(file)
        file.close()

        return content

    def load_dictionaries(self):

        word_to_idx = self.file_handling(filename="wsj_word_to_idx.pkl")
        pos_to_idx = self.file_handling(filename="wsj_pos_to_idx.pkl")

        return word_to_idx, pos_to_idx

    def file_preprocessing(self) -> Tuple[List, List]:

        def word_to_idx_pipeline(x):
            sample_to_idx = [self.word_to_idx[tok] for tok in x]
            return torch.tensor(sample_to_idx, dtype=torch.int64)
        def pos_to_idx_pipeline(x):
            tags_to_idx = [self.pos_to_idx[pos] for pos in x]
            return torch.tensor(tags_to_idx, dtype=torch.int64)

        sentences = treebank.tagged_sents()[:2700]
        list_of_lines, list_of_line_tags, samples_with_labels = [], [], []

        for sentence in sentences:

            current_line, current_tags = [], []
            for word_and_tags in sentence:
               current_line.append(word_and_tags[0])
               current_tags.append(word_and_tags[1])
            
            #padding
            if len(current_line) < 50:
                for i in range(0,50-len(current_line)):
                    current_line.append('PADDING')
                    current_tags.append('PADDING')

            #word_to_idx as tensors
            current_line_to_idx = word_to_idx_pipeline(current_line)
            current_tags_to_idx = pos_to_idx_pipeline(current_tags)
            list_of_lines.append(current_line_to_idx)
            list_of_line_tags.append(current_tags_to_idx)

        samples_with_labels.append(list(zip(list_of_lines, list_of_line_tags)))
        print(samples_with_labels)
        return samples_with_labels

    def __init__(self, myDataset=None):
        
        print("Preparing the dataset...")
        print("loading dictionaries...")
        self.word_to_idx, self.pos_to_idx = self.load_dictionaries()
        print("dictionary ready!")

        print("pre-processing the dataset...")
        self.samples = self.file_preprocessing()
        print("dataset ready!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

###############################
mydataset = DataLoader(dataset=WSJDataset(), batch_size=16, shuffle=False)

for idx,(sample) in enumerate(mydataset):
    if idx>0:
        break
    print(len(sample[0][0]))
    print(len(sample[0][1]))