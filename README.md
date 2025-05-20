# 🏁 diaCW – Deep Q-Learning Autonomous Racing Agent

This project implements a **Deep Q-Network (DQN)** to train an autonomous racing agent using the `CarRacing-v3` environment from Gymnasium. The model learns to drive around procedurally generated tracks using stacked visual inputs, reward shaping, and experience replay. It was developed for the COMP4105/COMP3004 Designing Intelligent Agents module at Newcastle University.

---

## 📦 File Structure

```
diaCW/
│
├── agent.py                 # DQN agent: architecture, replay, action selection
├── trainModel.py            # Main training loop and GIF logging
├── testModel.py             # Testing interface with model selector + replay
├── parameters.py            # All hyperparameter configs (batch size, ε, γ, etc.)
├── utilFunctions.py         # Preprocessing functions (grayscale, action mapping, stacking)
├── graphs.py                # Generates reward/loss/generalisation graphs
├── carRecolourInteractive.py# Tool for recolouring car sprite in GIFs
├── training_log.csv         # Training log of reward/loss per episode
│
├── models/                  # Folder containing trained models (.pth)
├── gifs/                    # Folder for saved replay GIFs
├── bestModels/              # High-scoring models from multi-agent training runs
└── README.md
```

---

## 🧠 Techniques Employed

- **Deep Q-Learning (DQN)**
  - 2D CNN for processing stacked grayscale input frames
  - Epsilon-greedy exploration with configurable decay and minimum
  - Discrete action mapping for low-dimensional control (e.g., forward, brake, left, right)

- **Experience Replay**
  - Transition buffer sampled randomly to break correlation
  - Mini-batch updates to stabilize convergence

- **Guided Warmup**
  - Initial episodes optionally guided with centerline bias to reduce early spin-outs

- **Reward Shaping**
  - Positive: forward speed, direction alignment, staying on track
  - Negative: drifting, reversing, inactivity

- **Model Checkpointing**
  - Best model auto-saved based on reward
  - GIF and model checkpoint saved every 100 episodes
  - Loss + reward logged to CSV for plotting

---

## 🧪 Experimental Variants

Three single DQN configurations were tested:

| Variant      | ε Decay | γ (Discount) | Batch | Memory | Notable Traits         |
|--------------|---------|--------------|--------|--------|------------------------|
| Original     | 0.999   | 0.95         | 64     | 5000   | Stable but lower avg   |
| Improved     | 0.997   | 0.95         | 32     | 10000  | Fastest convergence, best generalisation |
| Experimental | 0.996   | 0.97         | 48     | 15000  | High volatility, strong early growth |

A comparison against **Dueling DQN** was also conducted. Despite its theoretical advantages, Single DQN outperformed it in average reward, consistency, and generalisability under CPU-only constraints.

---

## 📚 Libraries Used

| Library       | Purpose                                      |
|---------------|----------------------------------------------|
| `gymnasium`   | CarRacing-v3 simulation environment          |
| `pygame`      | Rendering and manual control support         |
| `torch`       | Neural network and gradient-based learning   |
| `opencv-python` (`cv2`) | Frame preprocessing (grayscale, resize) |
| `numpy`       | Array ops and buffer management              |
| `imageio`     | GIF generation from replay frames            |
| `csv`         | Training log writer                          |
| `datetime`, `os` | Timestamping and model path logic          |

---

## 🚦 How to Run

### 1. Install Dependencies
```bash
pip install torch numpy opencv-python pygame imageio gymnasium[box2d]
```

### 2. Train the Agent
```bash
python trainModel.py
```
- Trains the agent over a configurable number of episodes (default: 1000)
- Saves models to `models/`
- Logs performance to `training_log.csv`
- GIFs saved for episode 0, final episode, and every 100th

### 3. Test an Agent
```bash
python testModel.py
```
- Select `.pth` model interactively
- Renders test drive and exports replay to `gifs/`

---

## 📊 HUD Visual Output

| Element         | Description                              |
|------------------|------------------------------------------|
| `0006` text       | Total cumulative reward (episode score) |
| White bar         | Car speed (forward velocity)            |
| Green bar         | Steering input (left/right)             |
| Red bar           | Drift offset (momentum vs heading)      |
| Blue bars         | Per-wheel acceleration (LF, RF, LR, RR) |

---

## 🖼️ Customisation Tools

- **`carRecolourInteractive.py`**  
  Use this tool to recolour the car sprite in GIFs (default car is red). Supports interactive file selection and HSV masking to isolate car body vs. kerbs.

---

## 📈 Analysis & Evaluation

All reward/loss logs are saved to `training_log.csv` and can be plotted with `graphs.py`. Comparison graphs available include:

- Training reward curves (raw + moving average)
- Frequency of high-scoring episodes (>800, >900)
- Generalisation performance across 30 track seeds

> 📍 Figures referenced in the final report include:  
> `singleDQNvsDoubleDQN.png`, `highRewardScoreFreqComparison.png`, `randomGeneralisationTestingComparison.png`

---

## 🔧 Future Enhancements

- Add continuous-action agents (DDPG, SAC, PPO)
- Improve long-term memory with LSTM integration
- Apply reward shaping using centerline or off-track penalty maps
- Expand evaluation to multi-agent competitive scenarios

---

## 📂 Coursework Context

This repository was developed for the COMP4105 (MEng) or COMP3004 (BSc) module *Designing Intelligent Agents* at Newcastle University. The final report evaluates RL architecture selection, parameter tuning, and generalisation behaviour in procedural track navigation.

---

## 🧑‍💻 Author

**edmbell** – 2025 Coursework Submission

---

## ⚙️ Conda Environment Setup (Mac Example)

```bash
conda create -n racecar-dqn python=3.10
conda activate racecar-dqn
brew install swig  # macOS-only for Box2D

pip install torch numpy opencv-python pygame imageio
pip install "gymnasium[box2d]"  # may fallback to pybox2d
```
