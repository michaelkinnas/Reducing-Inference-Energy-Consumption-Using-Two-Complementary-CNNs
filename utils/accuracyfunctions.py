from torch import argmax, round, sigmoid, float

# Multiclass accuracy function
def multi_accuracy_fn(preds, true):
    # print((argmax(preds, dim=1) == true).type(float).sum().item())
    return (argmax(preds, dim=1) == true).type(float).sum().item()
    
# Binary accuracy function
def bin_accuracy_fn(preds, true):
    return (round(sigmoid(preds)) == true).type(float).sum().item()
