from config import text_cleaner,white_space_cleaner
with open("reddit_jokes.txt","r",encoding="utf-8") as f:
    text = f.read()

n = 10000
data = text[:n]


#Converting data into sentences
sentences = data.split("\n")
clean_sentence = []
#remove empty sequence
count = 0
while count != len(sentences):
    if text_cleaner(sentences[count]) == '':
        count+=1
    else:
        clean_sentence.append(text_cleaner(sentences[count]))
        count+=1


#tokenizing every sentences 

sentence_list = []
for x in range(len(clean_sentence)):
    word_list = []
    temp_word = ''
    count = 0
    for i in clean_sentence[x]:
        count +=1
        if (i == ' ') | (count == len(clean_sentence[x])):
            if i == clean_sentence[x][-1]:
                temp_word+=str(i)
            word_list.append(white_space_cleaner(temp_word))
            temp_word = ' '
        else:
            temp_word+=str(i)
        
    sentence_list.append(word_list)

vocab = {'<UNK>':0}
for i in range(len(sentence_list)):
    for token in sentence_list[i]:
        if token not in vocab:
            vocab[token] = len(vocab)
# Finding pairs
class Pairs:
    
    def __init__(self,sentence_list):
        self.sentence_list = sentence_list
        self.pairs = {}

    def forward(self):
        self.pairs = {}
        for i in range(len(self.sentence_list)):
            for x in range(len(self.sentence_list[i])):
                if self.sentence_list[i][x] == sentence_list[i][-1]:
                    continue
                else:
                    for z,y in zip([self.sentence_list[i][x]],[self.sentence_list[i][x+1]]):
                        if (z,y) not in self.pairs.keys():
                            self.pairs[(z,y)] = 1
                        else:
                            self.pairs[(z,y)] += 1
        
        return self.pairs
    
# Add new token to vocab
class AddToken:

    def __init__(self,pairs,vocab,encod_iter):
        self.pairs = pairs
        self.vocab = vocab
        self.max_value = 0
        self.encod_iter = encod_iter

    def forward(self):

        if (self.pairs == {}) | (max(self.pairs.values()) == 3):
            return (f"encoding_iter: {self.encod_iter} , max value in pair counts is 3 (minvalue) ")
        
        else:
            self.max_value = max(self.pairs.values())
        
        for i in self.pairs.keys():
            if self.pairs[i] == self.max_value:
                if i not in self.vocab.keys():
                    temp = ''.join([x for x in i])
                    self.vocab[temp] = len(self.vocab)
                    # print(f"Element adding to vocab: {temp}")
        # print(f"Lenth of vocabulary: {len(self.vocab)}")
        return self.vocab
                
# Updating new tokens to token list

class UpdateTokenList:

    def __init__(self,sentence_list,vocab,encoding_iter):
        self.sentence_list = sentence_list
        self.vocab = vocab
        self.encoding_iter = encoding_iter
        self.warn = ''
    def forward(self):
        for i in range(len(self.sentence_list)):
            int_del = ''
            for x in range(len(self.sentence_list[i])):
                if x + 1 == len(self.sentence_list[i]):
                    continue
                else:
                    if ''.join((self.sentence_list[i][x],self.sentence_list[i][x+1])) in self.vocab.keys():
                        self.sentence_list[i][x] = self.sentence_list[i][x] + self.sentence_list[i][x+1]
                        int_del = x
            if int_del == '':
                continue
            else:
                if self.sentence_list[i][int_del] == self.sentence_list[i][-1]:
                    continue
                else:
                    if self.sentence_list[i][int_del] in self.vocab.keys():
                        # print(f"Word deleting: {self.sentence_list[i][int_del+1]}")
                        del(self.sentence_list[i][int_del+1])
        return self.warn , self.sentence_list
    
#Byte_Pair_Encoder
class BPEWord:
    def __init__(self,sentence_list,vocab,encod_iter):
        self.sentence_list = sentence_list
        self.vocab = vocab
        self.encoding_iter = encod_iter
        self.warn = ''

    def forward(self):
        for i in range(self.encoding_iter):
            #building adjacent pairs
            pairs = Pairs(self.sentence_list)
            out_pairs = pairs.forward()

            # Add token to vocab
            addtok = AddToken(out_pairs,self.vocab,i)
            # getting our new vocab
            temp_vocab = addtok.forward()
            if isinstance(temp_vocab,str):
                break
            else:
                self.vocab = temp_vocab

            # Updating our token list
            update_tok = UpdateTokenList(self.sentence_list,self.vocab,i)
            self.warn , self.sentence_list = update_tok.forward()
            # print(f"Encoding_iter: {i}")

        return self.vocab,self.sentence_list


# Testing
BpairEncode = BPEWord(sentence_list=sentence_list,vocab=vocab,encod_iter=3)
new_vocab,new_sentence_list = BpairEncode.forward()
