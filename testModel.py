# python testModel.py

import os
import random
import imageio
import matplotlib.pyplot as plt
from datetime import datetime
import gymnasium as gym
import pygame
from collections import deque
from agent import Agent
from utilFunctions import *
from parameters import *

# ---- settings ----
saveGif = True
os.makedirs("gifs", exist_ok=True)

# ---- model selection prompt ----
modelDir = "models"
modelFiles = [f for f in os.listdir(modelDir) if f.endswith(".pth")]

if not modelFiles:
    raise FileNotFoundError("No .pth model files found in the models directory.")

print("Available Models:")
for idx, fileName in enumerate(modelFiles):
    print(f"[{idx}] {fileName}")

selectedIdx = input(f"Select a model index [0-{len(modelFiles) - 1}]: ")
try:
    selectedIdx = int(selectedIdx)
    modelPath = os.path.join(modelDir, modelFiles[selectedIdx])
except (ValueError, IndexError):
    raise ValueError("Invalid selection. Please enter a valid index number.")

# ---- test mode selection ----
print("\nChoose test mode:")
print("[1] Single test with custom seed")
print("[2] Run on 10 random seeds (save top 3 and worst GIFs)")

mode = input("Enter mode (1 or 2): ")

# ---- environment and agent setup ----
env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")

a = Agent(frameStackNum, actionSpace, learningRate, memorySize, trainingBatchSize, discountFactor, epsilon, epsilonDecay, epsilonMin)
a.loadModel(modelPath)
print(f"✅ Loaded model: {modelPath}")

def run_episode(seed):
    state, _ = env.reset(seed=seed)
    state = rgbToGray(state)
    frameQueue = deque([state] * frameStackNum, maxlen=frameStackNum)
    frames = []
    totalReward = 0

    while True:
        frameStack = getFrameStack(frameQueue)
        action = a.getAction(frameStack)
        realAction = mapDiscreteAction(action)

        for _ in range(frameStackNum):
            state, reward, terminated, truncated, _ = env.step(realAction)
            totalReward += reward
            state = rgbToGray(state)
            frameQueue.append(state)

            frame = env.render()
            if saveGif and frame is not None:
                frames.append(frame)

            if terminated or truncated:
                break
        if terminated or truncated:
            break

    return totalReward, frames, seed

if mode == "1":
    seed = int(input("Enter seed for the test run (e.g., 42): "))
    reward, frames, seedUsed = run_episode(seed)

    if saveGif and frames:
        baseModelName = os.path.splitext(os.path.basename(modelPath))[0]
        timeStamp = datetime.now().strftime("%H-%M-%S")
        gifPath = f"gifs/test_{baseModelName}_seed{seed}_{timeStamp}.gif"
        imageio.mimsave(gifPath, frames, duration=1/30)
        print(f"🎞️ Test run saved as GIF: {gifPath}")
        print(f"📈 Total Reward: {reward:.2f}")

elif mode == "2":
    results = []
    for i in range(10):
        seed = random.randint(0, 9999)
        reward, frames, seedUsed = run_episode(seed)
        results.append((reward, frames, seedUsed))
        print(f"Run {i + 1}/10 | Seed: {seedUsed} | Reward: {reward:.2f}")

    results.sort(reverse=True, key=lambda x: x[0])
    best3 = results[:3]
    worst1 = results[-1:]

    for idx, (reward, frames, seedUsed) in enumerate(best3 + worst1):
        label = f"top{idx + 1}" if idx < 3 else "worst"
        baseModelName = os.path.splitext(os.path.basename(modelPath))[0]
        timeStamp = datetime.now().strftime("%H-%M-%S")
        gifPath = f"gifs/{label}_{baseModelName}_seed{seedUsed}_{timeStamp}.gif"
        imageio.mimsave(gifPath, frames, duration=1/30)
        print(f"💾 Saved {label} run GIF: {gifPath} | Reward: {reward:.2f}")
else:
    print("❌ Invalid mode selected.")
