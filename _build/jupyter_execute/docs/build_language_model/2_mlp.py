#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import string
import numpy as np

import torch
import torch.nn.functional as F
from torch import Tensor

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from dataclasses import dataclass, field

torch.set_printoptions(linewidth=120)
seed = 2**31 - 1


# <hr>
# 
# # Multilayer Perceptron 
# <hr>

# In this notebook, we will develop a [multilayer perceptron](https://www.wikiwand.com/en/Multilayer_perceptron) (MLP) model to predict the next character in a sequence. Our approach is based on the paper ["A Neural Probabilistic Language Model" by Bengio et al. (2003)](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf),which presents a word-level language model. However, we will work with characters instead of words and follow a similar modeling approach. The goal is to train an MLP to learn the patterns and structure of a character sequence and use this knowledge to accurately predict the next character in the sequence.
# 
# ```{figure} ./media/2_mlp_2022-10-26-19-49-45.png
# :width: 400px
# :name: MLP
# 
# Multilayer Perceptron with embeddings (Bengio et al. 2003)
# ```
# 
# 
# 
# 

# ## Setup

# **Load the data**

# In[2]:


# Load the data
with open('./data/names.txt', 'r') as f:
    names = f.read().splitlines()
print(f'Total # names: {len(names):.>25,}')
print(f"Total # characters in all names: {len(''.join(names)):,}")
print(names[:5])


# **Encoding and decoding characters**   
# We define an encoder `ch2ix` and decoder `ix2ch` function that maps a character with a numerical representation (i.e. a unique integer) and vice-versa. 

# In[3]:


chars = '.' + string.ascii_lowercase
ch2ix = {s: i for i, s in enumerate(chars)}
ix2ch = {i: s for s, i in ch2ix.items()}

print('Encoder ch2ix:\n', ch2ix)
print('Decoder ix2ch:\n', ix2ch)
print('E.g. emma:', [ch2ix[c] for c in '.emma.'])


# **Dataset**

# In[4]:


@dataclass
class Dataset:
    """
    Attributes:
    names: list of names.
    block_size: context length: how many characters do we take to predict the next one?
    """

    names: list[str]
    block_size: int = 3
    verbose: bool = False
    X: Tensor = field(init=False)
    Y: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self._make_dataset()

    def _make_dataset(self) -> None:
        X, Y = [], []
        for w in self.names:
            if self.verbose:
                print(w)
            context = [0] * self.block_size
            for ch in w + '.':
                ix = ch2ix[ch]
                X.append(context)
                Y.append(ix)
                if self.verbose:
                    print(''.join(ix2ch[i] for i in context), '--->', ix2ch[ix])
                context = context[1:] + [ix]  # crop and append
        self.X = torch.tensor(X)
        self.Y = torch.tensor(Y)


# In[5]:


ds = Dataset(names[:1], block_size=3, verbose=True)
X, Y = ds.X, ds.Y


# ## Character embedding

# To train the MLP to predict the next character in a sequence, we will represent each input character as a learnable m-dimensional vector. These vectors will be initialized randomly and will be adjusted during the training process. Characters that are similar to each other, such as vowels, will tend to be located closer to each other in this vector space.   
# 
# To embed our 27 characters into an m-dimensional space, we can create a 27xm lookup table (C) and randomly initialize it. To obtain the embedding vector for the i-th character, we can either encode the character as a one-hot vector and multiply it by C, or simply use the index to retrieve the i-th row of C. This will give us the m-dimensional vector that represents the i-th character, which can be used as input to the MLP model.

# **one-hot vs index**

# In[6]:


i = 5
C = torch.randn((27, 2))
# One-hot method
onehot = F.one_hot(torch.tensor(i), num_classes=27).float()
print(onehot @ C)
# Index method
print(C[i])


# Using the index is much faster than using a one-hot vector.

# In[7]:


get_ipython().run_line_magic('timeit', 'onehot = F.one_hot(torch.tensor(i), num_classes=27).float()')
get_ipython().run_line_magic('timeit', 'index = C[i]')


# We can also index by a multi-dimentional tensor.

# In[8]:


t = torch.tensor([[5, 5, 1], [2, 3, 4]])
print(t.shape)
print(C[t])
print(C[t].shape)


# So to embed simultaniously all the integers of X, we simply do `C[X]`

# In[9]:


print(X.shape)
print(C[X].shape)


# **Plot untrained embedding**

# In[10]:


C = torch.randn((27, 2))
plt.figure(figsize=(6, 6))
plt.scatter(C[:, 0], C[:, 1], s=200, color='lightblue')
for i in range(27):
    plt.text(C[i, 0].item(), C[i, 1].item(), chars[i], ha="center", va="center", color='black')
plt.axis('equal')
plt.title('Untrained embeddings')
plt.show()


# ## Building MLP step by step

# We will build a MLP model has three components:
# 1. An **embedding matrix** `C` that maps each input character to an m-dimensional embedding vector
# 2. **A hidden layer** that applies the tanh non-linearity to the output of the concatenated embedding vectors
# 3. An **output layer** that produces logits from the output of the hidden layer and applies the softmax function to generate probabilities
# 
# The hidden layer has:
# - Weights `W1` with shape (`block_size` * `emb_dim`, `hid_dim`)
# - Bias b1 with shape (`hid_dim`)
# 
# The output layer has:
# - Weights `W2` with shape (`hid_dim`, 27)
# - Bias `b2` with shape (27)
# 
# 
# 

# **Hyperparameters**

# In[11]:


block_size = X.shape[1]  # 3 (# input characters)
emb_dim = 2
hid_dim = 100


# **Embedding**
# 

# In[12]:


C = torch.randn((27, emb_dim))  # (27, emb_dim)
emb = C[X]  # (bs, block_size, emb_dim)
print(emb.shape)
print(emb)


# To perform the matrix multiplication between the embedding tensor  (`bs`, `block_size`, `emb_dim`) and the weights `W1` (`block_size` * `emb_dim`, `hid_dim`) we must concatenate the blocks of the embedding tensor. This can be achieved using the [torch.cat()](https://pytorch.org/docs/stable/generated/torch.cat.html) or with [torch.Tensor.view()](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html).
# 
# ```{note}
# `torch.Tensor.view()` is much more efficient, because unlike `torch.cat()`, no memory is being changed, created, copied or moved  by the view() operation. The storage is unchanged.
# A good blogpost on how pytorch internaly works to stores tensors:[PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/).
# ```
# 

# In[13]:


get_ipython().run_line_magic('timeit', 'emb.view(-1, block_size*emb_dim)')
get_ipython().run_line_magic('timeit', 'torch.cat(torch.unbind(emb, dim=1), dim=1)')


# Concatenate embeddings with view

# In[14]:


emb_concat = emb.view(-1, block_size * emb_dim)
print(emb_concat.shape)
print(emb_concat)


# **Hidden layer**

# In[15]:


W1 = torch.randn(block_size * emb_dim, hid_dim)
b1 = torch.randn(hid_dim)


# We can now calculate the hidden states h.
# 
# ```{note}
# The shape of `b1` is (hid_dim). We will add it to a tensor with shape (`block_size` * `emb_dim`, `hid_dim`). In order to do the addition `b1` will be broadcasted into a tensor of the same shape as `W1`. In other words, the bias will be added to all the rows of the tensor `emb` x` W1`.
# ```

# In[16]:


h = emb_concat @ W1 + b1
print(h.shape)
print(h)


# Add tanh non-linear activation

# In[17]:


x = torch.linspace(-10, 10, 100)
y = torch.tanh(x)

plt.figure(figsize=(3, 2))
plt.plot(x, y)
plt.title('tanh')
plt.show()


# In[18]:


h = torch.tanh(h)  # (bs, hid_dim)
print(h.shape)
print(h)


# **Output layer**

# In[19]:


W2 = torch.randn(hid_dim, 27)
b2 = torch.randn(27)


# In[20]:


logits = h @ W2 + b2  # (bs, 27)
print(logits.shape)
print(logits)


# Transform logits into probabilities with softmax.

# In[21]:


counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)  # (bs, 27)
print(probs.shape)
print(probs)


# ## Loss function

# To calculate the loss of the model, we extract the predicted probability of the true label and compute the negative log-likelihood. The objective is to minimize this loss in order to improve the model's ability to predict the correct character in the sequence.    
# Btw: Minimizing the negative log-likelihood is equivalent to maximizing the likelihood.  

# **Negative log-likelihood loss (NLL)**

# In[22]:


loss = -probs[torch.arange(X.shape[0]), Y].log().mean()
loss


# **Cross entropy loss**: The negative log-likelihood and cross entropy are two interpretations of the same formula.   
# 
# $\text{cross entropy}= H(p,q)= -\sum_x{p(x)log(q(x))}$   
# with   
# $x$: $\in$ classes   
# $p(x)$: true probability distribution  
# $q(x)$: predicted probability distribution
# 
# There are many good reasons to use `F.cross_entropy()` instead of manually calculating the loss. It's much more efficient and it's more numerically well behaved. The manual calculation uses `logits.exp()` and that can lead to overflow when an element of logits is very high. `F.cross_entropy()` avoids these kinds of problems.

# In[23]:


loss = F.cross_entropy(logits, Y)
loss


# ## MLP model

# In[24]:


class MLP:
    def __init__(self, block_size: int = 3, emb_dim: int = 2, hid_dim: int = 100, seed: int | None = None) -> None:
        g = torch.Generator().manual_seed(seed) if seed else None
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.C = torch.randn((27, emb_dim), generator=g)
        self.W1 = torch.randn((block_size * emb_dim, hid_dim), generator=g)
        self.b1 = torch.randn(hid_dim, generator=g)
        self.W2 = torch.randn((hid_dim, 27), generator=g)
        self.b2 = torch.randn(27, generator=g)
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True
        self.n_parameters = sum(p.nelement() for p in self.parameters)

    def __call__(self, xs: Tensor) -> Tensor:
        """Does a forward pass and returns the logits"""
        emb = self.C[xs].view(-1, self.block_size * self.emb_dim)
        h = emb @ self.W1 + self.b1
        h = torch.tanh(h)
        logits = h @ self.W2 + self.b2
        return logits

    def zero_grad(self) -> None:
        """Sets the gradients to None"""
        for p in self.parameters:
            p.grad = None

    def optimise_step(self, lr) -> None:
        """Updat the parameters"""
        for p in self.parameters:
            p.data += -lr * p.grad


# ### Training 1 batch

# In[25]:


ds = Dataset(names, block_size=3)
X, Y = ds.X, ds.Y
Xsamp, Ysamp = X[:32], Y[:32]
model = MLP(block_size=3, hid_dim=100)
print(f'# parameters: {model.n_parameters}')


# In[26]:


emb_dim = 3
hid_dim = 100
model = MLP(block_size=3, emb_dim=emb_dim, hid_dim=hid_dim)
print(f'# parameters: {model.n_parameters}')


# In[27]:


losses = []
loss = 0
for _ in range(20000):
    logits = model(Xsamp)
    loss = F.cross_entropy(logits, Ysamp)
    losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    model.optimise_step(0.1)
print(f'{loss=}')


# The low loss value of 0.2515 when training a single batch suggests that the model is overfitting, as it has a large number of parameters (3,481) relative to the small number of samples (32). This allows the model to memorize the true labels, rather than learning generalizable patterns.
# 
# It is also important to note that the loss will not be able to decrease to zero because there are multiple different y values for the same x values in the batch. For example, the 5 samples with all [0, 0, 0] x values have different y values. In this case, the minimum loss that can be achieved when the network has memorized all samples can be calculated as follows:
# 
# $H(p,q) = -\sum_x{p(x)log(q(x))}$   
# where:   
# $p(x)=1$ and $q(x)=1$ for all "unique" samples (not [0, 0, 0])   
# $p(x)=1$ and $q(x)=1/5$ for the 5 "non-unique" samples ([0, 0, 0])   
# therefore:   
# $H(p,q) = -\frac{5\log(1/5)}{32}=0.2515$ 
# 
# 

# ### Training the full dataset

# In[28]:


ds = Dataset(names, block_size=3)
X, Y = ds.X, ds.Y
model = MLP(block_size=3, emb_dim=2, hid_dim=100)
print(f'# parameters: {model.n_parameters}')


# In[29]:


losses = []
loss = 0

for _ in range(1000):
    logits = model(X)
    loss = F.cross_entropy(logits, Y)
    losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    model.optimise_step(0.1)
print(f'{loss=}')


# In[30]:


plt.figure(figsize=(6, 3))
plt.plot(np.log(losses))
plt.title('log-loss full dataset')
plt.xlabel('steps')
plt.ylabel(r'$log_{10}(loss)$')
plt.show()


# Training for 1000 iterations on the full dataset is very slow. It takes more than 2 min to obtain a loss of 2.5610.

# ### Training with mini-batches

# Instead of training on the full dataset, we can randomly select at each iteration a portion of the dataset, a mini-batch,  and do a forward and backward pass and update on each mini-batch. 

# In[31]:


ds = Dataset(names, block_size=3)
X, Y = ds.X, ds.Y
model = MLP(block_size=3, hid_dim=100, seed=seed)
print(f'# parameters: {model.n_parameters}')


# In[32]:


def get_minibatch(X: Tensor, Y: Tensor, bs: int = 32, g: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    idx = torch.randint(0, X.shape[0], (bs,), generator=g)
    return X[idx], Y[idx]


bs = 320
n = 1000
losses = []
loss = 0
g = torch.Generator().manual_seed(seed)
for _ in range(n):
    Xb, Yb = get_minibatch(X, Y, bs, g)
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    model.optimise_step(0.3)
print(f'Minibatch loss: {loss.item():.4f}')  # type: ignore


# In[33]:


plt.figure(figsize=(6, 3))
plt.plot(range(len(losses)), np.log(losses))
ma = 50
losses_ma = np.convolve(losses, np.ones(ma) / ma, mode='valid')
plt.plot(range(ma // 2, len(losses_ma) + ma // 2), np.log(losses_ma))
plt.title('log-loss mini-batches')
plt.xlabel('steps')
plt.ylabel(r'$log_{10}(loss)$')
plt.show()


# The use of minibatches in training can result in a decrease in the precision of the gradient calculation. While the gradient calculated from minibatches may not be as accurate as the gradient obtained from training on the entire dataset, it is still sufficient for practical use. It is generally more efficient to make progress with an approximate gradient by taking more steps, rather than obtaining an exact gradient but taking fewer steps. To accurately gauge the performance of the network, it is necessary to evaluate the loss on the entire dataset.

# In[34]:


logits = model(X)
loss = F.cross_entropy(logits, Y)
print(f'{loss=}')


# Training on mini-batches of size 320 is much faster than training on the full dataset. Training for 1000 iterations takes less than 1 sec and has a similar loss of 2.5756 compared to the loss of 2.5610 from training on the full dataset. 

# ## Train | valid | test split

# If the number of parameters of a model is large enough then the model has enough capacity to memorize the training set. The loss will be very low, but if you sample from it then it will only produce samples it already saw in the trainingset. It won't produce new data.   
# If you evaluate the loss on withheld names, i.e. samples that are not in the traininset, than that loss on those samples will be very high. The model is **overfitted**.  
# 
# The solution is to split the dataset into 3 sets: **training set, validation set and test set**. The training set (~80%) is for optimizing the parameters of the model. the validation set (~10%) is used to tune the hyperparameters of the model and the test set (~10%) is used to test the quality of the model at the very end. 

# In[35]:


def train_valid_test_split(
    names: list, train_frac: float = 0.8, valid_frac: float = 0.1, seed: None | int = None
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Function that split the dataset.
    It returns a tuple of (Xtrn, Ytrn, Xval, Yval, Xtst, Ytst)
    """
    if seed:
        random.seed(seed)
    words = names.copy()
    n = len(words)
    n1 = int(train_frac * n)
    n2 = n1 + int(valid_frac * n)
    random.shuffle(words)
    ds = Dataset(words[:n1])
    Xtrn, Ytrn = ds.X, ds.Y
    ds = Dataset(words[n1:n2])
    Xval, Yval = ds.X, ds.Y
    ds = Dataset(words[n2:])
    Xtst, Ytst = ds.X, ds.Y
    return Xtrn, Ytrn, Xval, Yval, Xtst, Ytst


# In[36]:


ds = Dataset(names, block_size=3)
X, Y = ds.X, ds.Y
Xtrn, Ytrn, Xval, Yval, Xtst, Ytst = train_valid_test_split(names, seed=seed)


# ## Tuning hyperparameters

# ### Learning rate

# One way to determin a reasonable learning rate is as follows: 
# - Determine lower and upper bounds for the learning rate through trial and error. A learning rate that is too low may result in slow convergence, while a learning rate that is too high may result in unstable or oscillating loss.
# - Train the network for one step using the lower bound learning rate. At each subsequent step, exponentially increase the learning rate up to the upper bound.
# - Plot the learning rate (lr) against the loss. A reasonable learning rate is likely to be at the plateau of the plot.

# **Upper and lower bound**

# In[37]:


fix, axs = plt.subplots(2, 2, figsize=(6, 3))

bs = 320
n = 100
model = MLP(block_size=3, hid_dim=100)

for i, lr in enumerate([1e-4, 1e-3, 1e0, 1e1]):
    losses = []
    loss = 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        Xb, Yb = get_minibatch(X, Y, bs, g)
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        losses.append(loss.item())
        model.zero_grad()
        loss.backward()
        model.optimise_step(lr)
    ax = axs.flatten()  # type: ignore
    ax[i].plot(losses)
    ax[i].set_title(lr)
plt.tight_layout()


# **Traing with exponentially increasing lr**

# In[38]:


lb, ub = 0.001, 1
n = 1000
bs = 32
lr_exp = torch.linspace(np.log10(lb), np.log10(ub), n)  # [-3 ... 0]
lrs = 10**lr_exp
g = torch.Generator().manual_seed(seed)
model = MLP(block_size=3, emb_dim=2, hid_dim=100, seed=seed)

losses = []
for i, lr in enumerate(lrs):
    Xb, Yb = get_minibatch(X, Y, bs, g)
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    model.optimise_step(lr)


# In[39]:


ma = 100
losses_ma = np.convolve(losses, np.ones(ma) / ma, mode='valid')

plt.figure(figsize=(5, 2.5))
plt.plot(lr_exp, losses, alpha=0.2)
plt.plot(lr_exp[ma // 2 - 1 : -ma // 2], losses_ma)
min_ix = np.argmin(losses_ma)
min_lr = lrs[min_ix + ma // 2].item()
plt.axvline(np.log10(min_lr), ls='--', c='r')
plt.text(np.log10(min_lr) + 0.1, 20, f'lr: {min_lr:.2}')

plt.ylabel('loss')
plt.xlabel(r'$\log_{10}(lr)$')
plt.show()


# A reasonably good learning rate is somewhere between $10^{-1}$ and $10^{-0.5}$. 
# 
# **Weight decay**: We train our model with the decent learning rate till loss doesn't decrease anymore. Then we decay the learning rate with factor 10 and train a little longer.

# In[40]:


def calc_loss(model: MLP, X, Y) -> float:
    logits = model(X)
    loss = F.cross_entropy(logits, Y)
    return loss.item()


lr = 0.15
bs = 320
n = 500_000
decay_step = 495_000
batch_losses = []
train_losses = []
valid_losses = []
loss = 0

g = torch.Generator().manual_seed(seed)
model = MLP(block_size=3, emb_dim=2, hid_dim=100, seed=seed)

lrd = False
for i in range(n):
    # Lr decay
    if i > decay_step and not lrd:
        lr /= 10
        lrd = True
        print(f'Lr decay: {lr}')
    Xb, Yb = get_minibatch(X, Y, bs, g)
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    batch_losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    model.optimise_step(lr)
    if i % 100 == 0:
        train_loss = calc_loss(model, Xtrn, Ytrn)
        train_losses.append(train_loss)
        valid_loss = calc_loss(model, Xval, Yval)
        valid_losses.append(valid_loss)
    if i % 1000 == 0:
        train_loss = calc_loss(model, Xtrn, Ytrn)
        valid_loss = calc_loss(model, Xval, Yval)
        print(f'Loss (train | valid) : {i:>7} | {train_loss:.4f} |  {valid_loss:.4f}')

print(f'Loss (total | test) : {calc_loss(model, X, Y):.4f} | {calc_loss(model, Xtst, Ytst):.4f}')


# In[41]:


plt.figure(figsize=(5, 2.5))

plt.plot(train_losses[100:], alpha=0.5)
plt.plot(valid_losses[100:], alpha=0.5)
plt.legend(['train loss', 'valid loss'])
plt.title('train / valid loss')
plt.show()


# We obtain a total loss of 2.17, which is a significant improvement from the loss of 2.45 of the bigram model!

# ### Hidden dim

# During previous training, the training set loss was 2.1874 and the validation set loss was 2.1930. This suggests that overfitting is not occurring, and therefore it may be beneficial to increase the model size (i.e., number of parameters).
# 
# To identify the optimal hidden layer size, we will train the network using different configurations and monitor the total and validation loss. It is expected that the network will overfit with high values of hidden dimension.

# In[42]:


lr = 0.15
bs = 320
n = 100_000
decay_step = 90_000

for i, hd in enumerate([50, 100, 200, 300, 400]):
    loss = 0
    g = torch.Generator().manual_seed(seed)
    model = MLP(block_size=3, emb_dim=2, hid_dim=hd, seed=seed)
    lrd = False
    for j in range(n):
        # Lr decay
        if j > decay_step and not lrd:
            lr /= 10
            lrd = True
        Xb, Yb = get_minibatch(X, Y, bs, g)
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        model.zero_grad()
        loss.backward()
        model.optimise_step(lr)
    print(f'hdim: {hd}, #paramters: {model.n_parameters}')
    print(f'Loss hdim={hd} (total | valid) : {calc_loss(model, X, Y):.4f} | {calc_loss(model, Xval, Yval):.4f}')


# Increasing the width of the hidden layer results in worse validation loss and does not improve performance. This may suggest that the true bottleneck is not the hidden layer, but rather the size of the embeddings, or that the larger network requires more training steps to optimize its increased number of parameters.

# ### Embedding dim

# In[43]:


lr = 0.15
bs = 320
n = 200_000
decay_step = 190_000

for i, ed in enumerate([2, 3, 4, 5]):
    loss = 0
    g = torch.Generator().manual_seed(seed)
    model = MLP(block_size=3, emb_dim=ed, hid_dim=100, seed=seed)
    lrd = False
    for j in range(n):
        # Lr decay
        if j > decay_step and not lrd:
            lr /= 10
            lrd = True
        Xb, Yb = get_minibatch(X, Y, bs, g)
        logits = model(Xb)
        loss = F.cross_entropy(logits, Yb)
        model.zero_grad()
        loss.backward()
        model.optimise_step(lr)
    print(f'ed: {ed}, #paramters: {model.n_parameters}')
    print(f'Loss (total | valid) : {calc_loss(model, X, Y):.4f} | {calc_loss(model, Xval, Yval):.4f}')


# ## Visialize embeddings

# In[44]:


emb_dim = 2
hid_dim = 50
n = 60000
bs = 25000
lr = 0.3

ds = Dataset(names, block_size=3)
X, Y = ds.X, ds.Y
g = torch.Generator().manual_seed(seed)
model = MLP(block_size=3, emb_dim=emb_dim, hid_dim=hid_dim, seed=seed)
print(f'# parameters: {model.n_parameters}')
losses = []
loss = 0
Cs = []

for i in range(n):
    if i % 100 ==0:
        Cs.append(model.C.detach().clone().numpy())
    ixs = torch.randint(0, len(X), (bs,))
    Xb = X[ixs]
    Yb = Y[ixs]
    logits = model(Xb)
    loss = F.cross_entropy(logits, Y[ixs])
    losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    model.optimise_step(lr)
    if i % 1000 == 0:
        print(f'{i:>10} | {loss:.4f}')


# In[45]:


df = pd.DataFrame()
cols = [f'x{i}' for i in range(emb_dim)]
for i, emb in enumerate(Cs[:1000000]):
    emb = pd.DataFrame(emb, columns=cols)
    emb['ep'] = i
    emb['chr'] = list('.' + string.ascii_lowercase)
    emb.loc[emb['chr'].isin(list('aeiou')), 'col'] = 'r'
    emb.loc[~emb['chr'].isin(list('aeiou')), 'col'] = 'b'

    df = pd.concat([df, emb])
df = df.reset_index(drop=True)


# In[46]:


fig = px.scatter(
    df,
    x='x0',
    y='x1',
    text='chr',
    color='col',
    animation_frame='ep',
    range_x=[df['x0'].min()-.2, df['x0'].max()+.2],
    range_y=[df['x1'].min()-.2, df['x1'].max()+.2],
)

fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 5  # type: ignore
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 5  # type: ignore

fig.update_traces(marker={'size': 20, 'opacity': 0.35})
fig.update_layout(height=800, width=800)
fig


# The visualization of the embeddings shows that the representations for vowels are grouped together, with the special character "." being distinct from the other characters.

# ## Sampling names

# In[47]:


lr = 0.1
bs = 32
n = 200_000
decay_step = 100_000
loss = 0

g = torch.Generator().manual_seed(seed)
model = MLP(block_size=3, emb_dim=8, hid_dim=100, seed=seed)

lrd = False
for i in range(n):
    # Lr decay
    if i > decay_step and not lrd:
        lr /= 10
        lrd = True
    Xb, Yb = get_minibatch(X, Y, bs, g)
    logits = model(Xb)
    loss = F.cross_entropy(logits, Yb)
    batch_losses.append(loss.item())
    model.zero_grad()
    loss.backward()
    model.optimise_step(lr)
    if i % 10000 == 0:
        train_loss = calc_loss(model, Xtrn, Ytrn)
        valid_loss = calc_loss(model, Xval, Yval)
        print(f'lr: {lr} | Loss (train | valid) : {i:>7} | {train_loss:.4f} |  {valid_loss:.4f}')

print(f'Loss (total | test) : {calc_loss(model, X, Y):.4f} | {calc_loss(model, Xtst, Ytst):.4f}')


# In[48]:


g = torch.Generator().manual_seed(seed)

for i in range(20):

    out = [] 
    context = [0] * block_size # '...'
    while True:
        logits = model(torch.tensor(context))  # predict log-counts
        p = torch.softmax(logits, 1) 
        ix = torch.multinomial(p, num_samples=1, replacement=False, generator=g).item()
        out.append(ix2ch[ix])  # type: ignore
        context = context[1:] + [ix]
        if ix == 0:
            break
    print(''.join(out))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




