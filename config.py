import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "checkpoint.pt"
seed = 100

#parameters
n_embd = 384
dropout = 0.3

#functions
def text_cleaner(string1):
    string1 = string1.lower()
    string1 = string1.replace('(','')
    string1 = string1.replace(')','')
    string1 = string1.replace('?','')
    string1 = string1.replace("'","")
    string1 = string1.replace(".","")
    string1 = string1.replace("''","")
    string1 = string1.replace(":","")
    string1 = string1.replace("``","")
    string1 = string1.replace("!","")
    string1 = string1.replace("*","")
    string1 = string1.replace(";","")
    string1 = string1.replace('``','')
    string1 = string1.replace(',','')
    #Better use regex
    return string1

def white_space_cleaner(text):
    text = text.rstrip()
    return text


