import torch
from utils import new_sentence_list,new_vocab

#convert sentence into numerical indices sentence
class Text_to_indices:
    def __init__(self,new_sentence,new_vocab):
        self.sentence = new_sentence
        self.vocab = new_vocab
        self.numrical_sentence = []

    def forward(self):
        
        for i in range(len(self.sentence)):
            single_sentence = []
            for token in self.sentence[i]:
                if token in self.vocab:
                    single_sentence.append(self.vocab[token])
                else:
                    single_sentence.append(self.vocab['<UNK>'])
            self.numrical_sentence.append(single_sentence)
        return self.numrical_sentence

t_i = Text_to_indices(new_sentence=new_sentence_list,new_vocab=new_vocab)
numerical_sentences = t_i.forward()
#print(len(numerical_sentences),"numerical_sentence")
#build training sequence (1->2) (1,2->3)
training_sequence = []
for sentence in numerical_sentences:
    for i in range(1,len(sentence)):
        training_sequence.append(sentence[:i+1])

#print(len(training_sequence))

#padding-> adding zeros -> same length of each sequence
sentence_length = []
for sequence in training_sequence:
    sentence_length.append(len(sequence))

max_length = max(sentence_length)
#print(max_length)

padded_training_sequence = []
for sequence in training_sequence:
    padded_training_sequence.append([0]*(max_length - len(sequence)) + sequence)

# converting into tensor
padded_training_sequence = torch.tensor(padded_training_sequence,dtype=torch.long)


# split data into train and test
X = padded_training_sequence[:,:-1]
y = padded_training_sequence[:,-1]

vocab_size = len(new_vocab)

#print(X.shape,y.shape)

