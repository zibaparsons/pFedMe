import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
from FLAlgorithms.users.userbase import User

# Implementation for FedSRWADMM clients

class UserFedSRWADMM(User):
    def __init__(self, device, numeric_id, train_data, test_data, model, batch_size, learning_rate, beta, lamda,
                 local_epochs, optimizer):
        super().__init__(device, numeric_id, train_data, test_data, model[0], batch_size, learning_rate, beta, lamda,
                         local_epochs)

        if(model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def train(self, epochs, num_users): #num_users added by JLu
        LOSS = 0
        self.model.train()

        # Added by JLu
        param_z_old = {}
        param_x_old = {}
        for params in self.model.parameters():
            param_x_old = params.clone()
            param_z_old = params.clone()


        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            X, y = self.get_next_train_batch()
            self.optimizer.zero_grad()

            # Modified by jlu
            # output = self.model(X)
            output = self.local_model_x(X)

            loss = self.loss(output, y)
            loss.backward()

            # Commented by JLu
            # self.optimizer.step()
            # self.clone_model_paramenter(self.model.parameters(), self.local_model)

            # Added by JLu
            for param_x, param_y, param_z in zip(self.local_model_x.parameters(), self.model.parameters(), self.local_dual_z.parameters()):
                param_x_old.data = param_x.data
                param_z_old.data = param_z.data

                # Added by jlu
                param_x.data = param_y.data - self.learning_rate * (param_x.grad.data) #(param_z.data - param_x.grad.data)
                param_z.data = param_z.data + 1 / self.learning_rate * (param_x.data - param_y.data)
                param_y.data = param_y.data + 1/num_users * (param_x.data - param_x_old.data + self.learning_rate * self.beta * (param_z_old.data  - param_z.data))

                # modified by zp
                # param_x.data = param_y.data + self.learning_rate * (param_z.data - param_x.grad.data)
                # param_z.data = param_z.data + self.beta * (param_x.data - param_y.data)
                # param_y.data = param_y.data + (1/num_users) * (param_x.data - param_x_old.data + self.beta * (param_z_old.data - param_z.data))

        return LOSS

