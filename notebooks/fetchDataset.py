from typing import List, Tuple
from prepareDictionary import PennTreeBankDictionary
ds = PennTreeBankDictionary()
# from prepareDataset import PennTreeBankDataset
import torch
import pickle, gzip
from torch.utils.data import Dataset, DataLoader

class myDataset(Dataset):

    def compose_dictionaries(self): 

        word_to_idx, idx_to_word, pos_to_idx, idx_to_pos = ds.vocabulary()
        return word_to_idx, idx_to_word, pos_to_idx, idx_to_pos

    def file_parser(self,filename) -> Tuple[List,List]:

        filepath = "C:/Users/rahin/projects/WSJ-POS-tagger/data/processed/"+filename+'.pklz'
        samples_with_labels = []
        file = gzip.open(filepath,'rb')
        samples_with_labels = pickle.load(file)
        
        return samples_with_labels

    def file_tensor(self, sentences_and_tags) -> Tuple[List, List]:

        def token_pipeline(x):
            if len(x) < 50:
                for i in range(0,50-len(x)):
                    x.append('PADDING')
            return [self.word_to_idx[tok] for tok in x]

        def pos_pipeline(x):
            if len(x) < 50:
                for i in range(0,50-len(x)):
                    x.append('PADDING')
            return [self.pos_to_idx[pos] for pos in x]

        sent_to_idx, tags_to_idx = [], []
        for sent_tag in sentences_and_tags:
            if len(sent_tag[0]) >50:
                continue
            sent_to_idx.append(torch.tensor(token_pipeline(sent_tag[0])))
            tags_to_idx.append(torch.tensor(pos_pipeline(sent_tag[1])))

        return sent_to_idx, tags_to_idx

    def __init__(self, raw_dataset=None):

        print("STEP 01: Look-up Tables...")
        self.word_to_idx, self.idx_to_word, self.pos_to_idx, self.idx_to_pos = self.compose_dictionaries()
        print("dictionaries ready!")

        print("STEP 02: Fetching the dataset...")
        self.samples_with_labels = self.file_parser(raw_dataset)
        print("done!")

        print("STEP 03: Tokens and tags to Numbers...")
        self.samples_to_idx, self.labels_to_idx = self.file_tensor(self.samples_with_labels)

    def __len__(self):    
        return len(self.samples_to_idx)

    def __getitem__(self, index):
        return self.samples_to_idx[index], self.labels_to_idx[index]

# validation_dataset = DataLoader(dataset=myDataset("PennTreeBankValid")
#                                         ,shuffle=False
#                                         ,batch_size=16)

