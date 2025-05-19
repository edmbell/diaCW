# python testModel.py

import os
import random
import imageio
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
os.makedirs("generalisation_gifs", exist_ok=True)

# ---- model selection ----
modelDir = "bestModels"
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
print("[3] Run on 10 predefined control seeds (comparable testing)")

mode = input("Enter mode (1, 2 or 3): ")

# ---- environment and agent setup ----
env = gym.make("CarRacing-v3", continuous=True, render_mode="rgb_array")
a = Agent(frameStackNum, actionSpace, learningRate, memorySize, trainingBatchSize, discountFactor, epsilon, epsilonDecay, epsilonMin)
a.loadModel(modelPath)
print(f"✅ Loaded model: {modelPath}")

# ---- run a test episode ----
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

# ---- mode 1: single run ----
if mode == "1":
    seed = int(input("Enter seed for the test run (e.g., 42): "))
    reward, frames, seedUsed = run_episode(seed)
    if saveGif and frames:
        baseModelName = os.path.splitext(os.path.basename(modelPath))[0]
        timeStamp = datetime.now().strftime("%H-%M-%S")
        gifPath = f"gifs/test_{baseModelName}_seed{seed}_{timeStamp}.gif"
        imageio.mimsave(gifPath, frames, duration=1/30)
        print(f"🎞️ Saved GIF: {gifPath} | Reward: {reward:.2f}")

# ---- mode 2 and 3: batch generalisation test ----
elif mode in ["2", "3"]:
    results = []
    is_control = (mode == "3")
    predefined_seeds =  [89, 1297, 2371, 3209, 4561, 5021, 6301, 7351, 8597, 9631] # sophie germain primes
                       #[61, 1607, 2557, 3761, 4099, 5113, 6277, 7517, 8291, 9473] # euclidian primes
                        #[2, 1103, 2287, 3307, 4013, 5107, 6073, 7523, 8117, 9397] # balanced primes
                      #[173, 1741, 2857, 3187, 4157, 5387, 6883, 7547, 8089, 9547] # random primes
                        #[7, 1657, 2269, 3169, 4219, 5167, 6211, 7057, 8269, 9241] # cuban primes

    for i in range(10):
        seed = predefined_seeds[i] if is_control else random.randint(0, 9999)
        reward, frames, seedUsed = run_episode(seed)
        results.append((reward, frames, seedUsed))
        print(f"Run {i + 1}/10 | Seed: {seedUsed} | Reward: {reward:.2f}")

    results.sort(reverse=True, key=lambda x: x[0])
    best3 = results[:3]
    worst1 = results[-1:]

    # extract numeric suffix from model filename
    baseModelName = os.path.splitext(os.path.basename(modelPath))[0]
    suffix = ''.join(filter(str.isdigit, baseModelName)) or "00000"

    # save top 3 and worst
    for idx, (reward, frames, seedUsed) in enumerate(best3 + worst1):
        label = f"{idx + 1}" if idx < 3 else "L"
        filename = f"genVis{suffix}-{label}-{seedUsed}.gif"
        gifPath = os.path.join("generalisation_gifs", filename)
        imageio.mimsave(gifPath, frames, duration=1/30)
        print(f"💾 Saved {label} run GIF: {gifPath} | Reward: {reward:.2f}")

else:
    print("❌ Invalid mode selected.")
