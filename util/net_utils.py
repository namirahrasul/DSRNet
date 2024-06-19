# returns the total number of parameters in the model that require gradients.
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# model.parameters(): This method returns an iterator over all the parameters (weights and biases) in the model.
#p.numel(): This method returns the number of elements in the parameter tensor p.

def count_conv_layers(model): #counts total no. conv layers in a model
    cnt = 0
    for mo in model.modules(): #model.modules(): This method returns an iterator over all submodules (layers) of the model, including the model itself.
        if type(mo).__name__ == 'Conv2d': #type(mo).__name__: This retrieves the string name of the class to which the submodule mo belongs.
            cnt += 1
    #model, total number of conv layers and total paramters with gradient
    print(type(model).__name__, cnt, count_parameters(model))
