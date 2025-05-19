# python parameters.py

# total number of training episodes
episodeCount = 1000

# number of stacked frames used as state input
frameStackNum = 3

# discrete action indices used by agent and action mapper
actionSpace = [0, 1, 2, 3, 4]

# dqn training hyperparameters
learningRate =          0.001
memorySize =            10000# 15000# 10000# 5000
trainingBatchSize =     32# 48# 32# 64
discountFactor =        0.95# 0.97# 0.95

# epsilon-greedy exploration settings
epsilon =               1.0
epsilonDecay =          0.997# .996# .997# 0.999
epsilonMin =            0.05# 0.1# 0.05

# enable/disable live rendering during training
render = False
