# Intro

The notebooks are my notes taken when following the course by **Andrej Karpathy** on neural networks and language modelling. This course starts all the way at the basics. We implement a bigram character-level language model, which we will further complexify into a modern Transformer language model, like GPT.



## Lecture 1: Building makemore
![](https://img.shields.io/badge/Status-Finished-brightgreen)


Notebook: [](./1_bigrams.ipynb)

 In this video, the focus is on (1) introducing torch.Tensor and its subtleties and use in efficiently evaluating neural networks and (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification).

<iframe width="600" height="400" src="https://www.youtube.com/embed/PaCmpygFfXo" title="The spelled-out intro to language modeling: building makemore" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Lecture 2: MLP
![](https://img.shields.io/badge/Status-WIP-orange)

Notebook:

We implement a multilayer perceptron (MLP) character-level language model. In this video we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).

<iframe width="600" height="400" src="https://www.youtube.com/embed/TCH_1BHY58I" title="Building makemore Part 2: MLP" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Lecture 3: Activations & Gradients, BatchNorm
![](https://img.shields.io/badge/Status-Not_Started-red)

Notebook:

We dive into some of the internals of MLPs with multiple layers and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. We also look at the typical diagnostic tools and visualizations you'd want to use to understand the health of your deep network. We learn why training deep neural nets can be fragile and introduce the first modern innovation that made doing so much easier: Batch Normalization.

<iframe width="600" height="400" src="https://www.youtube.com/embed/P6sfmUTpUmc" title="Building makemore Part 3: Activations & Gradients, BatchNorm" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Lecture 4: Becoming a Backprop Ninja
![](https://img.shields.io/badge/Status-Not_Started-red)

Notebook:

We take the 2-layer MLP (with BatchNorm) from the previous video and backpropagate through it manually without using PyTorch autograd's loss.backward(). That is, we backprop through the cross entropy loss, 2nd linear layer, tanh, batchnorm, 1st linear layer, and the embedding table. Along the way, we get an intuitive understanding about how gradients flow backwards through the compute graph and on the level of efficient Tensors, not just individual scalars like in micrograd. This helps build competence and intuition around how neural nets are optimized and sets you up to more confidently innovate on and debug modern neural networks.

<iframe width="600" height="400" src="https://www.youtube.com/embed/q8SA3rM6ckI?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ" title="Building makemore Part 4: Becoming a Backprop Ninja" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
