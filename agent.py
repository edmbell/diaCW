# python agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import random
from datetime import datetime
import os

# single dqn agent model

class Agent:
    def __init__(self, frameStackNum, actionSpace, learningRate, memorySize, trainingBatchSize, discountFactor,
                 epsilon, epsilonDecay, epsilonMin):
        # set device: mps first (for mac), then cuda, else cpu
        if torch.backends.mps.is_available():
            print("MPS")
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            print("CUDA")
            self.device = torch.device("cuda")
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

        # init replay memory
        self.memory = deque(maxlen=memorySize)

        # init model and optimizer
        self.model = self.createModel().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learningRate, eps=1e-7)

        print(f"Agent initialized with {sum(p.numel() for p in self.model.parameters())} parameters.")


    def createModel(self) -> torch.nn.Module:
        # defines cnn-based deep q-network
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
        # selects action with highest predicted q-value
        with torch.no_grad():
            stateTensor = torch.tensor(stateStack, dtype=torch.float32, device=self.device).unsqueeze(0)
            qValues = self.model(stateTensor)
            return torch.argmax(qValues).item()

    def getActionExplore(self, stateStack):
        # epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actionSpace)
        return self.getAction(stateStack)

    def addToMemory(self, frameStack, action, reward, nextFrameStack, terminated):
        # stores transition in replay memory
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
        # trains model using random batch from memory
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

        # decay epsilon after update
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

        return loss.item()

# debug print if run standalone
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


""" # dueling dqn agent model

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def add(self, transition, td_error=1.0):
        max_prio = self.priorities.max() if self.buffer else td_error
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = prios / prios.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = abs(td_error) ** self.alpha

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, frameStackNum, actionSpace, learningRate, memorySize, trainingBatchSize, discountFactor,
                 epsilon, epsilonDecay, epsilonMin, targetUpdateFreq=1000):
        if torch.backends.mps.is_available():
            print("MPS")
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            print("CUDA")
            self.device = torch.device("cuda")
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
        self.targetUpdateFreq = targetUpdateFreq
        self.learnStepCounter = 0
        self.beta = 0.4  # importance-sampling beta

        self.memory = PrioritizedReplayBuffer(memorySize)

        self.model = self.createModel().to(self.device)
        self.target_model = self.createModel().to(self.device)
        self.updateTargetModel()

        self.optimizer = optim.Adam(self.model.parameters(), lr=learningRate, eps=1e-7)

        print(f"Agent initialized with {sum(p.numel() for p in self.model.parameters())} parameters.")

    def createModel(self) -> torch.nn.Module:
        class DuelingDQN(nn.Module):
            def __init__(self, inputChannels, numActions):
                super(DuelingDQN, self).__init__()
                self.feature = nn.Sequential(
                    nn.Conv2d(in_channels=inputChannels, out_channels=6, kernel_size=7, stride=3),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(in_channels=6, out_channels=12, kernel_size=4),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
                )
                self._to_linear = None
                self._get_flattened_size(inputChannels)

                self.fc_value = nn.Sequential(
                    nn.Linear(self._to_linear, 216),
                    nn.ReLU(),
                    nn.Linear(216, 1)
                )

                self.fc_advantage = nn.Sequential(
                    nn.Linear(self._to_linear, 216),
                    nn.ReLU(),
                    nn.Linear(216, numActions)
                )

            def _get_flattened_size(self, inputChannels):
                with torch.no_grad():
                    dummy = torch.zeros(1, inputChannels, 96, 96)
                    x = self.feature(dummy)
                    self._to_linear = x.view(1, -1).size(1)

            def forward(self, x):
                x = self.feature(x)
                x = x.view(x.size(0), -1)
                value = self.fc_value(x)
                advantage = self.fc_advantage(x)
                return value + advantage - advantage.mean(dim=1, keepdim=True)

        return DuelingDQN(self.frameStackNum, len(self.actionSpace))

    def updateTargetModel(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def getAction(self, stateStack):
        with torch.no_grad():
            stateTensor = torch.tensor(stateStack, dtype=torch.float32, device=self.device).unsqueeze(0)
            qValues = self.model(stateTensor)
            return torch.argmax(qValues).item()

    def getActionExplore(self, stateStack):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actionSpace)
        return self.getAction(stateStack)

    def addToMemory(self, frameStack, action, reward, nextFrameStack, terminated):
        # Use max error as initial priority
        with torch.no_grad():
            state = torch.tensor(np.expand_dims(frameStack, 0), dtype=torch.float32, device=self.device)
            next_state = torch.tensor(np.expand_dims(nextFrameStack, 0), dtype=torch.float32, device=self.device)
            q_val = self.model(state)[0, action].item()
            next_action = torch.argmax(self.model(next_state)).item()
            target_val = reward
            if not terminated:
                target_val += self.discountFactor * self.target_model(next_state)[0, next_action].item()
            td_error = abs(q_val - target_val)
        self.memory.add((frameStack, action, reward, nextFrameStack, terminated), td_error)

    def saveModel(self, path=None):
        os.makedirs("models", exist_ok=True)
        if path is None:
            timestamp = datetime.now().strftime("%H-%M-%S")
            path = f"models/agentModel({timestamp}).pth"
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def loadModel(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
        self.updateTargetModel()

    def replay(self):
        if len(self.memory) < self.trainingBatchSize:
            return 0

        miniBatch, indices, weights = self.memory.sample(self.trainingBatchSize, beta=self.beta)
        weights = weights.to(self.device)

        states = torch.tensor(np.stack([t[0] for t in miniBatch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t[1] for t in miniBatch], device=self.device)
        rewards = torch.tensor([t[2] for t in miniBatch], dtype=torch.float32, device=self.device)
        nextStates = torch.tensor(np.stack([t[3] for t in miniBatch]), dtype=torch.float32, device=self.device)
        terminated = torch.tensor([t[4] for t in miniBatch], dtype=torch.bool, device=self.device)

        qValues = self.model(states)
        currentQ = qValues.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            nextQ_online = self.model(nextStates)
            next_actions = torch.argmax(nextQ_online, dim=1, keepdim=True)
            nextQ_target = self.target_model(nextStates)
            targetQ = rewards + (~terminated) * self.discountFactor * nextQ_target.gather(1, next_actions).squeeze(1)

        td_errors = (currentQ - targetQ).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        loss = ((currentQ - targetQ) ** 2 * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

        self.learnStepCounter += 1
        if self.learnStepCounter % self.targetUpdateFreq == 0:
            self.updateTargetModel()

        return loss.item()

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

"""