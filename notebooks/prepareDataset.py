# Penn Treebank English Dataset Preparation
import pickle
import gzip

class PennTreeBankDataset():

    def load_file(self, filename:str): # given file names, returns each line
        file = open(filename)
        roh_daten = file.readlines()
        return roh_daten

    def preprocessing(self, file:str):
        #step 1: fetch file contents
        raw_dataset = self.load_file(file)

        #step 2: form sentences and corresponding POS tags:
        all_samples, all_labels = [], []
        for sample in raw_dataset:

            sample = sample.replace('\n','')           
            sentence_dirty = list(filter(None,sample.split(')'))) # spliiting on closing brackets and viewing a subset of the result 
            sentence_clean, tags_clean = [],[]
            
            for word in sentence_dirty:
                word = word.replace("(","")
                sentence_clean.append(word.split(' ')[-1])
                tags_clean.append(word.split(' ')[-2]) if len(word) > 2 else ''
            
            if len(sentence_clean) != len(tags_clean):
                print("Mismatch in no. of tokens in the line!")
                break
            
            # add these two to the big list
            all_samples.append(sentence_clean) 
            all_labels.append(tags_clean)

        if len(all_samples) != len(all_labels):
            print(f"Total no. of samples: {len(all_samples)} \nTotal no. of POS tags: {len(all_labels)}")
            print("Mismatch in no. of lines")
            exit

        return list(zip(all_samples,all_labels))

    def export_files(self): # one time to create datasets

        list_of_files = ["PennTreeBankTrain.pklz","PennTreeBankTest.pklz","PennTreeBankValid.pklz"]
        list_of_files_tags = ["PennTreeBankTrainPOS.pkl","PennTreeBankTestPOS.pkl","PennTreeBankValidPOS.pkl"]
        list_of_datasets = ["02-21.10way.clean.txt","23.auto.clean.txt","22.auto.clean.txt"]

        for final,pos,raw in zip(list_of_files,list_of_files_tags,list_of_datasets):
            # print(f"Processing: {raw} => {final} \t {pos}")
            dataset = self.preprocessing(file="C:/Users/rahin/projects/WSJ-POS-tagger/data/corpus/"+str(raw))

            with gzip.open('C:/Users/rahin/projects/WSJ-POS-tagger/data/processed/' +str(final), 'wb') as f:
                pickle.dump(dataset, f)
                f.close()

ds = PennTreeBankDataset()
a = ds.export_files()