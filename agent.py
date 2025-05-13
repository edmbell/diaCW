# python agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
from datetime import datetime
import os

class Agent:
    def __init__(self, frameStackNum, actionSpace, learningRate, memorySize, trainingBatchSize, discountFactor,
                 epsilon, epsilonDecay, epsilonMin):
        # set device: cuda, then mps, fallback to cpu
        if torch.cuda.is_available():
            print("CUDA")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("MPS")
            self.device = torch.device("mps")
        else:
            print("CPU")
            self.device = torch.device("cpu")

        self.actionSpace = actionSpace
        self.frameStackNum = frameStackNum
        self.trainingBatchSize = trainingBatchSize
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.epsilonDecay = epsilonDecay
        self.epsilonMin = epsilonMin

        # experience memory buffer
        self.memory = deque(maxlen=memorySize)

        # model and optimizer
        self.model = self.createModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learningRate, eps=1e-7)

        print(f"Agent initialized with {sum(p.numel() for p in self.model.parameters())} parameters.")

    def createModel(self) -> torch.nn.Module:
        # defines convolutional deep q-network
        model = nn.Sequential(
            nn.Conv2d(in_channels=self.frameStackNum, out_channels=6, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(432, 216),
            nn.ReLU(),
            nn.Linear(216, len(self.actionSpace))
        )
        return model

    def getAction(self, stateStack):
        # returns action with highest q-value
        with torch.no_grad():
            stateTensor = torch.tensor(stateStack, dtype=torch.float32, device=self.device).unsqueeze(0)
            qValues = self.model(stateTensor)
            return torch.argmax(qValues).item()

    def getActionExplore(self, stateStack):
        # epsilon-greedy exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actionSpace)
        return self.getAction(stateStack)

    def addToMemory(self, frameStack, action, reward, nextFrameStack, terminated):
        # stores experience in memory
        self.memory.append((frameStack, action, reward, nextFrameStack, terminated))

    def saveModel(self, path=None):
        # saves model weights to file
        os.makedirs("models", exist_ok=True)
        if path is None:
            timestamp = datetime.now().strftime("%H-%M-%S")
            path = f"models/agentModel({timestamp}).pth"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def loadModel(self, path):
        # loads model weights from file
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

    def replay(self):
        # trains the model using a minibatch of past experiences
        miniBatch = random.sample(self.memory, self.trainingBatchSize)

        states = torch.tensor(np.stack([t[0] for t in miniBatch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t[1] for t in miniBatch], device=self.device)
        rewards = torch.tensor([t[2] for t in miniBatch], dtype=torch.float32, device=self.device)
        nextStates = torch.tensor(np.stack([t[3] for t in miniBatch]), dtype=torch.float32, device=self.device)
        terminated = torch.tensor([t[4] for t in miniBatch], dtype=torch.bool, device=self.device)

        qValues = self.model(states)
        with torch.no_grad():
            nextQValues = self.model(nextStates)

        chosenQValues = qValues.gather(1, actions.unsqueeze(1)).squeeze(1)
        targetQ = rewards + (~terminated) * self.discountFactor * torch.max(nextQValues, dim=1).values

        criterion = nn.MSELoss()
        loss = criterion(chosenQValues, targetQ)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        # decay epsilon after training step
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

        return loss.item()


# quick model printout for debugging
if __name__ == "__main__":
    dummyAgent = Agent(
        frameStackNum=4,
        actionSpace=[0, 1, 2],
        learningRate=0.001,
        memorySize=1000,
        trainingBatchSize=32,
        discountFactor=0.99,
        epsilon=1.0,
        epsilonDecay=0.995,
        epsilonMin=0.05
    )
    print(dummyAgent.model)
