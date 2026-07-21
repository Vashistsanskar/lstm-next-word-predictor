import torch
from model import LSTMModel
from config import *
from utils import white_space_cleaner,text_cleaner
from preparing import vocab_size,new_vocab,Text_to_indices,max_length
torch.manual_seed(seed)


#create model
model  =  LSTMModel(vocab_size=vocab_size,dropout=dropout).to(device)

# Load Trained checkpoint
checkpoint = torch.load(
    checkpoint,
    map_location = device,
    weights_only = True
)

model.load_state_dict(checkpoint["model"])
model.eval()

print(f"Loaded checkpoint from iteration {checkpoint['iter']}")

# Generate text
def tokenization(text):
        text = text_cleaner(text)
        sentence_list = []
        word_list = []
        temp_word = ''
        count = 0
        for i in text:
            count +=1
            if (i == ' ') | (count == len(text)):
                if i == text[-1]:
                    temp_word+=str(i)
                word_list.append(white_space_cleaner(temp_word))
                temp_word = ' '
            else:
                temp_word+=str(i)
            
        sentence_list.append(word_list)
        return sentence_list

def prediction(model,vocab,text):
     
     #tokenize
     tokenized_text = tokenization(text)

     #text-> numerical_indices
     t_i = Text_to_indices(new_sentence=tokenized_text,new_vocab=new_vocab)
     numerical_text = t_i.forward()
     len_numerical = len(numerical_text[0])
     #padding_text 
     padded_text = torch.tensor(([0] *(max_length - len_numerical)) + numerical_text[0],dtype=torch.long).unsqueeze(0)
     padded_text = padded_text.to(device)

     # send to model
     output = model(padded_text)

     #predicted index
     value,index = torch.max(output,dim=1)

     #merge with text
     return text + " " + list(vocab.keys())[index]

num_tokens = 5
input_text = "How are you bro"
for i in range(num_tokens):
     output_text = prediction(model,new_vocab,input_text)
     print(output_text)
     input_text = output_text



     