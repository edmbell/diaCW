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
renderMode = "rgb_array" if saveGif else "human"  # use offscreen rendering when saving gifs

os.makedirs("gifs", exist_ok=True)  # ensure gifs folder exists
os.makedirs("models", exist_ok=True)  # ensure models folder exists

# -------- logging setup --------
bestReward = float("-inf")  # track highest reward across episodes

logPath = "training_log.csv"
with open(logPath, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "TotalReward", "FinalLoss"])  # init csv headers

# -------- environment and agent setup --------
env = gym.make("CarRacing-v3", continuous=True, render_mode=renderMode)  # init racing env
agent = Agent(frameStackNum, actionSpace, learningRate, memorySize, trainingBatchSize, discountFactor, epsilon, epsilonDecay, epsilonMin)  # init dqn agent

for episode in range(episodeCount):
    seed = 42 if useFixedTrack else None  # 7 21 42 49 27 81
    state, _ = env.reset(seed=seed)
    state = rgbToGray(state)  # preprocess first frame

    frameQueue = deque([state] * frameStackNum, maxlen=frameStackNum)  # init frame stack
    frames = []  # store gif frames if needed

    totalReward = 0
    negativeRewardCounter = 0
    timeStep = 0

    while True:
        frameStack = getFrameStack(frameQueue)  # stack frames for state input
        action = agent.getActionExplore(frameStack)  # epsilon-greedy action selection
        realAction = mapDiscreteAction(action)  # convert to env-friendly action

        reward = 0
        for _ in range(frameStackNum):  # repeat same action across stack interval
            state, r, terminated, truncated, _ = env.step(realAction)
            reward += r
            state = rgbToGray(state)  # preprocess new frame
            frameQueue.append(state)

            frame = env.render()
            if saveGif and frame is not None and (episode == 0 and saveGifFirst or episode == episodeCount - 1 and saveGifLast):
                frames.append(frame)  # capture frames for gif

            if terminated or truncated:
                break  # exit inner loop if episode ends

        if timeStep > 100 and reward < 0:
            negativeRewardCounter += 1  # track persistent poor performance

        nextFrameStack = getFrameStack(frameQueue)
        agent.addToMemory(frameStack, action, reward, nextFrameStack, terminated)  # store experience

        if terminated or truncated or negativeRewardCounter > 25:
            print(f"Episode: {episode} | Total Reward: {totalReward:.2f} | Negative Reward Counter: {negativeRewardCounter}")

            if totalReward > bestReward:
                bestReward = totalReward
                agent.saveModel("models/best_model.pth")  # checkpoint best model
                print(f"✅ New best model saved with reward: {totalReward:.2f}")

            # save every 100th model + gif replay
            if episode % 100 == 0:
                timestamp = datetime.now().strftime("%H%M%S")
                modelPath = f"models/model{episode}_{timestamp}.pth"
                agent.saveModel(modelPath)

                if frames:
                    gifPath = f"gifs/replay{episode}_{timestamp}.gif"
                    imageio.mimsave(gifPath, frames, duration=1/30)  # save episode replay
                    print(f"💾 Saved model + GIF for episode {episode}")
            break  # end episode

        loss = 0
        if len(agent.memory) > trainingBatchSize:
            loss = agent.replay()  # train on batch from memory

        totalReward += reward
        print(f"Episode: {episode} | Time Step: {timeStep} | Action: {action} | Reward: {reward:.2f} | Loss: {loss:.5f}")

        timeStep += 1

    with open(logPath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([episode, totalReward, loss])  # log episode results

    if saveGif and frames and ((episode == 0 and saveGifFirst) or (episode == episodeCount - 1 and saveGifLast)):
        timeStamp = datetime.now().strftime("%H-%M-%S")
        gifPath = f"gifs/episode_{episode}_({timeStamp}).gif"
        imageio.mimsave(gifPath, frames, duration=1/30)  # save first/last episode gif
        print(f"Saved episode GIF: {gifPath}")

