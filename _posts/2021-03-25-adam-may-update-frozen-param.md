---
layout: post
title:  "Pytorch Adam may update frozen parameters"
date:   2021-03-25 20:12:00 
tags: deep learning, Pytorch
categories: coding
---

I was working on a deep learning training task that needed to freeze part of the parameters after 10 epochs of training. With Adam optimizer, even if I set

```Python
for parameter in model:
    parameter.requires_grad = False
``` 

There are still trivial differences before and after each epoch of training on those frozen parameters, like one can be from 0.1678 to 0.1674.

According to [this post](https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/15), Pytorch indeed has such an issue. 

A better practice is to split the training process into two parts like a pretraining-fine tuing work. In my situation, 

 * in the first part, train all the parameters and save them; (pretraining)

```python
model = Model()
dataloader = ...
checkpoint_path = ...

optim = torch.Adam(model.parameters(), lr=...)

for i in range(10):
    train(model, dataloader)

torch.save(model.state_dict(), checkpoint_path)

for p in model.parameters():
    if need_freeze(p):
        p.requires_grad = False
``` 

 * in the second part, reinitiate the model, start with a new optimizer without the frozen parameters, load the trained parameters; (fine tuning)

```python
model = Model()

"""
load saved model dict
"""

dataloader = ...
checkpoint_path = ...

filtered_params =  filter(lambda p: p.requires_grad, model.parameters())

optim = torch.Adam(filtered_params, lr=...)

for i in range(TOTAL_NUM - 10):
    train(model, dataloader)

torch.save(model.state_dict(), checkpoint_path)
``` 

* to check if the parameters are indeed not updated:

```python
model = Model()
dataloader = ...
checkpoint_path = ...

filtered_params =  filter(lambda p: p.requires_grad, model.parameters())

optim = torch.Adam(filtered_params, lr=...)

for i in range(TOTAL_NUM - 10):
    freeze_params = [p.clone() for p in self.model.parameters() if need_freeze(p)]

    train(model, dataloader)

    updated_params = [p.clone() for p in self.model.parameters() if need_freeze(p)]

    for i in range(len(freeze_params)):
        eq_tensor = torch.eq(freeze_params[i], updated_params[i])
        assert eq_tensor.all().data


torch.save(model.state_dict(), checkpoint_path)
```