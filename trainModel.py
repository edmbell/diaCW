"""
Used the following at cmd line to revert to Python 3.10 so the box2d install woudl work:

conda create -n racecar-dqn python=3.10
conda activate racecar-dqn
pip install torch numpy opencv-python pygame imageio
brew install swig               # for macOS
pip install "gymnasium[box2d]"  # or fallback: pip install pybox2d
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu #macOS speedup
pip install opencv-python

python trainModel.py
"""

# python trainModel.py

import pygame
from pygame.locals import *
import gymnasium as gym

from agent import Agent
from utilFunctions import *
from parameters import *

import numpy as np
from collections import deque
from datetime import datetime
import os
import imageio
import csv

# -------- config --------
render = True
saveGif = True
saveGifFirst = True
saveGifLast = True
useFixedTrack = False
renderMode = "rgb_array" if saveGif else "human"

os.makedirs("gifs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# -------- logging setup --------
bestReward = float("-inf")

logPath = "training_log.csv"
with open(logPath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "TotalReward", "FinalLoss"])

# -------- environment and agent setup --------
env = gym.make("CarRacing-v3", continuous=True, render_mode=renderMode)
agent = Agent(frameStackNum, actionSpace, learningRate, memorySize, trainingBatchSize, discountFactor, epsilon, epsilonDecay, epsilonMin)

for episode in range(episodeCount):
    seed = 42 if useFixedTrack else None
    state, _ = env.reset(seed=seed)
    state = rgbToGray(state)

    frameQueue = deque([state] * frameStackNum, maxlen=frameStackNum)
    frames = []

    totalReward = 0
    negativeRewardCounter = 0
    timeStep = 0

    while True:
        frameStack = getFrameStack(frameQueue)
        action = agent.getActionExplore(frameStack)
        realAction = mapDiscreteAction(action)

        reward = 0
        for _ in range(frameStackNum):
            state, r, terminated, truncated, _ = env.step(realAction)
            reward += r
            state = rgbToGray(state)
            frameQueue.append(state)

            frame = env.render()
            if saveGif and frame is not None and (episode == 0 and saveGifFirst or episode == episodeCount - 1 and saveGifLast):
                frames.append(frame)

            if terminated or truncated:
                break

        if timeStep > 100 and reward < 0:
            negativeRewardCounter += 1

        nextFrameStack = getFrameStack(frameQueue)
        agent.addToMemory(frameStack, action, reward, nextFrameStack, terminated)

        if terminated or truncated or negativeRewardCounter > 25:
            print(f"Episode: {episode} | Total Reward: {totalReward:.2f} | Negative Reward Counter: {negativeRewardCounter}")

            if totalReward > bestReward:
                bestReward = totalReward
                agent.saveModel("models/best_model.pth")
                print(f"✅ New best model saved with reward: {totalReward:.2f}")

            # save every 100th model + gif replay
            if episode % 100 == 0:
                timestamp = datetime.now().strftime("%H%M%S")
                modelPath = f"models/model{episode}_{timestamp}.pth"
                agent.saveModel(modelPath)

                if frames:
                    gifPath = f"gifs/replay{episode}_{timestamp}.gif"
                    imageio.mimsave(gifPath, frames, duration=1/30)
                    print(f"💾 Saved model + GIF for episode {episode}")
            break

        loss = 0
        if len(agent.memory) > trainingBatchSize:
            loss = agent.replay()

        totalReward += reward
        print(f"Episode: {episode} | Time Step: {timeStep} | Action: {action} | Reward: {reward:.2f} | Loss: {loss:.5f}")

        timeStep += 1

    with open(logPath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, totalReward, loss])

    if saveGif and frames and ((episode == 0 and saveGifFirst) or (episode == episodeCount - 1 and saveGifLast)):
        timeStamp = datetime.now().strftime("%H-%M-%S")
        gifPath = f"gifs/episode_{episode}_({timeStamp}).gif"
        imageio.mimsave(gifPath, frames, duration=1/30)
        print(f"Saved episode GIF: {gifPath}")
