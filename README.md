
# 🏁 diaCW – Deep Q-Learning Autonomous Racing Agent

This project implements a **Deep Q-Network (DQN)** to train an autonomous racing agent using the `CarRacing-v3` environment from Gymnasium. The model learns to drive around procedurally generated tracks using vision-based state inputs, reward shaping, and experience replay.

---

## 📦 File Structure

```
diaCW/
│
├── agent.py           # DQN agent class with replay, model creation, and action logic
├── parameters.py      # All training/testing hyperparameters in one place
├── trainModel.py      # Main training loop with frame stacking, reward shaping, and logging
├── testModel.py       # Run saved models with rendering and GIF export
├── utilFunctions.py   # Preprocessing functions: grayscale conversion, frame stack, action mapping
├── training_log.csv   # CSV file storing episode rewards and loss over time
│
├── models/            # Directory where trained .pth models are saved
├── __pycache__/       # Auto-generated Python bytecode cache
```

---

## 🧠 Techniques Employed

- **Deep Q-Learning (DQN)**
  - 2D convolutional layers with stacked grayscale image input
  - Epsilon-greedy exploration strategy with decay
  - Targeted action space for discrete car control (turning, acceleration, braking)

- **Reward Shaping**
  - Rewards for forward motion, checkpoint progression
  - Penalties for off-track drift, collision, and reversal
  - Encouragement for correct turning and momentum alignment

- **Guided Training Warmup**
  - Early episodes use centerline-based policy to reduce spinouts in RWD
  - Allows model to learn corner handling with stable behavior

- **Experience Replay**
  - Memory buffer for state-action transitions
  - Mini-batch sampling during training for stable gradient updates

- **Model Checkpointing**
  - Best model is saved during training
  - Every 100th episode saves both model weights and a replay GIF
  - Test script supports interactive model selection from saved files

---

## 📚 Libraries Used

| Library       | Purpose                                     |
|---------------|---------------------------------------------|
| `gymnasium`   | CarRacing-v3 simulation environment         |
| `pygame`      | Environment rendering & manual control      |
| `torch`       | Deep learning model and training framework  |
| `numpy`       | Numerical operations and data formatting    |
| `cv2` (OpenCV)| Frame preprocessing (grayscale + resize)    |
| `imageio`     | GIF generation from rendered test episodes  |
| `csv`         | Logging of training progress                |
| `os` / `datetime` | File and directory management            |

---

## 🚦 How to Run

### 1. Install dependencies
```bash
# recommended python version: 3.10
pip install torch numpy opencv-python pygame imageio gymnasium[box2d]
```

### 2. Train the Agent
```bash
python trainModel.py
```
- Models are saved to `models/`
- Progress is logged in `training_log.csv`
- Replays for episode 0, last episode, and every 100th episode saved as GIFs in `gifs/`

### 3. Test a Saved Model
```bash
python testModel.py
```
- You will be prompted to choose a `.pth` model from the `models/` directory.
- A replay GIF will be saved to `gifs/` for the test run.

---

## 🧪 Visual Output Key (HUD)

As seen in the CarRacing-v3 HUD:

| Indicator        | Meaning                                  |
|------------------|-------------------------------------------|
| `0006`           | Current reward score (total cumulative)   |
| White bar        | Speed indicator                           |
| Blue bars        | Wheel speeds (LF, LR, RF, RR)             |
| Green bar        | Steering input direction                  |
| Red bar          | Directional drift/momentum offset         |

---

## 📌 Notes

- The car uses a **rear-wheel drive (RWD)** model, making early training unstable without guided episodes.
- The environment uses procedurally generated tracks unless a fixed seed is set.
- Best and milestone (100th) models are saved to manage storage without losing training history.

---

## 🔧 Next Improvements

- Extend to continuous action space (e.g., DDPG or PPO)
- Evaluate generalization over randomized tracks
- Introduce Lidar-style raycasting inputs for better spatial awareness
- Multi-agent or competitive racing simulation

---

## 🧑‍💻 Author

**edmbell** – Private Coursework Submission


---

## ⚙️ Environment Setup (MacOS/Linux Example)

Used the following commands to set up a Python 3.10 environment for compatibility with `gymnasium[box2d]` and related libraries:

```bash
# create and activate environment
conda create -n racecar-dqn python=3.10
conda activate racecar-dqn

# install essential libraries
pip install torch numpy opencv-python pygame imageio

# macOS only: install swig for Box2D bindings
brew install swig

# install gymnasium with Box2D environment
pip install "gymnasium[box2d]"

# optional: improve PyTorch performance on macOS
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If `gymnasium[box2d]` fails to install, try the fallback:
```bash
pip install pybox2d
```

---

