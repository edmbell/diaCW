# python testModel.py

import gymnasium as gym
import pygame
from agent import Agent
from utilFunctions import *
from parameters import *
from collections import deque
from datetime import datetime
import imageio
import os

# ---- settings ----
saveGif = True  # to save the run as gif
fixedTrack = True  # fixed track seed for reproducibility when comparing training build models
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

# ---- environment and agent setup ----
env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")

# load trained agent with selected model
a = Agent(frameStackNum, actionSpace, learningRate, memorySize, trainingBatchSize, discountFactor, epsilon, epsilonDecay, epsilonMin)
a.loadModel(modelPath)
print(f"✅ Loaded model: {modelPath}")

# ---- initialize state ----
seed = 27 if fixedTrack else None # 42
state, _ = env.reset(seed=seed)
state = rgbToGray(state)
frameQueue = deque([state] * frameStackNum, maxlen=frameStackNum)

frames = []

# ---- run the episode ----
while True:
    frameStack = getFrameStack(frameQueue)
    action = a.getAction(frameStack)
    realAction = mapDiscreteAction(action)

    for _ in range(frameStackNum):
        state, reward, terminated, truncated, _ = env.step(realAction)
        state = rgbToGray(state)
        frameQueue.append(state)

        frame = env.render()
        if saveGif and frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    if terminated or truncated:
        break

# ---- save gif output ----
if saveGif and frames:
    timeStamp = datetime.now().strftime("%H-%M-%S")
    baseModelName = os.path.splitext(os.path.basename(modelPath))[0]
    gifPath = f"gifs/test_{baseModelName}_{timeStamp}.gif"
    imageio.mimsave(gifPath, frames, duration=1/30)
    print(f"🎞️ Test run saved as GIF: {gifPath}")
