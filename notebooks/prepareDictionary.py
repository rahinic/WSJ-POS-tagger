from prepareDataset import PennTreeBankDataset
ds = PennTreeBankDataset()
class PennTreeBankDictionary():

    def load_corpus(self):

        print("preparing train/test/valid datasets")
        valid_ds = ds.preprocessing(file="data/corpus/22.auto.clean.txt")
        test_ds = ds.preprocessing(file="data/corpus/23.auto.clean.txt")
        train_ds = ds.preprocessing(file="data/corpus/02-21.10way.clean.txt")
        complete_ds = valid_ds + test_ds + train_ds
        print("done")

        return complete_ds

    def tokens_and_tags(self):

        #Step 1: fetch dataset
        dataset = self.load_corpus()
        tokens, tags = [], []

        #Step 2: split sentences and pos tags to two separate list
        for sample in dataset:
            tokens.append(sample[0])
            tags.append(sample[1])

        #Step 3: list of lists to a single flat list and take unique words and pos tags
        all_tokens = [item for sublist in tokens for item in sublist]
        all_tokens = list(set(all_tokens))
        all_tags = [item for sublist in tags for item in sublist]
        all_tags = list(set(all_tags))

        return all_tokens, all_tags

    def vocabulary(self):
        
        print("preparing look-up dictionaries")
        words_in_corpus, pos_tags_in_corpus = self.tokens_and_tags()
        words_in_corpus.append('PADDING')
        pos_tags_in_corpus.append('PADDING')
        word_to_idx, pos_to_idx = {}, {}
        idx_to_word, idx_to_pos = {}, {} # for reverse look-up

        for idx, word in enumerate(words_in_corpus):
            word_to_idx[word] = idx
            idx_to_word[idx] = word

        for idx,pos in enumerate(pos_tags_in_corpus):
            pos_to_idx[pos] = idx
            idx_to_pos[idx] = pos

        print("done!")
        print(f"Total words in the dictionary: {len(word_to_idx)} \nTotal POS tags in the dictionary: {len(pos_to_idx)}")
        return word_to_idx, idx_to_word, pos_to_idx, idx_to_pos

# ex = PennTreeBankDictionary()
# a,b,c,d = ex.vocabulary()
# print(list(a.items())[:10])
# print(list(c.items())[:10])
