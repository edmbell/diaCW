# python parameters.py

# number of frames to stack for state input
frameStackNum = 3

# discrete action indices (must match agent's action mapping)
actionSpace = [0, 1, 2, 3, 4]

# dqn training hyperparameters
learningRate = 0.001
memorySize = 5000
trainingBatchSize = 64
discountFactor = 0.95

# epsilon-greedy exploration parameters
epsilon = 1.0
epsilonDecay = 0.999
epsilonMin = 0.05

# number of training episodes
episodeCount = 1000

# toggle rendering during training or testing
render = False
