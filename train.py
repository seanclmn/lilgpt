with open("input.txt", "r", encoding="utf-8") as f:
  text = f.read()
  


chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from strings to integers (this is part of our tokenization process)
stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}

# encoding and decoding functions. lambda is a shorthand for writing functions inline 
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])


# Now we encode the entire text dataset and store it into a torch.Tensor

import torch
# dtype = "data type": what kind of number every element in a tensor is stored as.
# A tensor uses one type for all its elements, which keeps it fast and memory-efficient.
#
# torch.long = 64-bit integer (whole numbers only), another name for torch.int64.
#
# Why torch.long: character IDs are labels, not quantities. They're used as indexes
# into the embedding table (one vector per character), and nn.Embedding requires
# integer inputs. Cross-entropy loss also expects class-index targets to be torch.long.
#
# Common dtypes:
#   torch.long (int64)    whole numbers    token IDs, indexes, class labels
#   torch.float (float32) decimals         model weights, embeddings, most calculations
#   torch.bool            True/False       masks
#
# Check a tensor's type with data.dtype (prints torch.int64 here).
data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8

example_train_data = data[:block_size]


torch.manual_seed(1337)
batch_size = 4 # number of parallel calculations we can run
block_size = 8 # block size

def get_batch(split):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  print(ix)
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+1+block_size] for i in ix])
  return x,y

xb, yb = get_batch("train")

import torch.nn as nn
import torch.nn.functional as F 

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    # We create a table of learnable numbers, self.token_embedding_table. 
    # We use it to look up rows in it by index
    # It is a matrix of weights
    # THey are at first random numbers, then altered by training
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    
  # nn.module has a __call__ function, which calls any declared function named "forward", which in python is automatically
  # called when used like:
  # model = BigramLanguageModel(vocab_size)
  # xb, yb = get_batch('train')
  # logits, loss = model(xb, yb)
  
  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)
    if targets is None:
      return logits, None
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
    # Loss is one number that says how wrong the model’s predictions were on a batch. 
    # Lower is better. Training is just repeatedly nudging the weights to make this number smaller.
    # Cross entropy is the specific formula used to calculate that number. 
    # It’s the standard choice whenever the model is picking one answer out of several options, 
    # like picking the next character out of 65.
    return logits, loss
  
model = BigramLanguageModel(vocab_size)
xb, yb = get_batch('train')
logits, loss = model(xb, yb)

print(model)