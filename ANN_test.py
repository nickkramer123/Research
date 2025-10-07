# import artificialNeuralNetwork as ANN
# from deap import base, creator, tools
# import random
# from sklearn.feature_selection import mutual_info_regression
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# from torch import nn




# # Step 1: with X, gather f(x)

# # using cpu
# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


# # import rossler attractor file and convert to pyTorch Tensor
# with open("./Research/data/values.pkl", "rb") as file: # Read as numpy arrays shape [3000, 3] (3000 samples, 3 features)
#     values = pickle.load(file) # 3000 lists of 3 objects [x1, y1, z1]

# values_array = np.array(list(values.values()), dtype=np.float32).reshape(-1, 3)
# values = torch.tensor(values_array).to(device)


# # get Neural network to turn 3 values into 1 value
# # returns V, or f(x) + e
# # class Network():
# #     # create a model
# #     model = ANN.NeuralNetwork().to(device)
# #     x = torch.tensor([[1.0, 2.0, 3.0]], device = device)
# #     y = model(x)
# #     print("Output 1: ", y.item())


# # # TODO get nueral network to work on 1 set of X from attractor
# # # returns V, or f(x) + e
# # class Network():
# #     # create a model
# #     model = ANN.NeuralNetwork().to(device)
# #     x = torch.tensor(values[0].unsqueeze(0), device = device)
# #     y = model(x)
# #     print("Output 2: ", y.item())



# # TODO get nueral network to loop through all of the values in big X
# # returns V, or f(x) + e
# class Network():
#     def __init__(self):
#         self.model = ANN.NeuralNetwork().to(device)

#     # TODO should values be given? probably
#     def run(self):
#         V_values = [] # initialize list to store V, of f(x) + e
#         for i in range(len(values)): # for every little x [x, y, x] value in big X
#             x = values[i].unsqueeze(0)
#             with torch.no_grad():  # no gradients needed
#                 y = self.model(x)
#             V_values.append(y.item())
#         return V_values



# net = Network()
# V = net.run()
# print(V)


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor






# TODO check difference made cus 16
class NeuralNetwork(nn.Module): # define neuralnetwork class that inherits from nn.Module
    # define layers of the function in initialization
    def __init__(self): 
        super().__init__()
        self.flatten = nn.Flatten() # Initialize the flatten layer to convert into contiguous array of values 
        self.linear_relu_stack = nn.Sequential( # ordered container of modules
            nn.Linear(3, 16), # appies a linear transformation on the input using stored weights and bases
            nn.ReLU(), # applies nonlinearity
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )

    # specify how data will pass through the network
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    



    
    
    
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from torchviz import make_dot


model = NeuralNetwork()

# torchviz
dummy_input = torch.randn(16, 3)
output = model(dummy_input)

dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("simple_net_graph", format="png", view=True) # Saves as png and opens it


# TODO tensorborad
# dummy_input = torch.randn(16, 3) # Create a dummy input tensor

# # Set up TensorBoard writer
# writer = SummaryWriter('runs/simple_nn_tensorboard')
# writer.add_graph(model, dummy_input)
# writer.close()






