# 🚀 Reinforcement Learning: Multi-Armed Bandit Experiments

This repository contains **modular implementations** of **multi-armed bandit algorithms**, inspired by Sutton & Barto’s *Reinforcement Learning: An Introduction*. The project is structured to support **parameter tuning, experimentation, and visualization** of different bandit algorithms.

---

## 📂 Project Structure

```
reinforcement_learning/
│── reinforcement_learning/  # Main package folder
│   ├── __init__.py
│   ├── bandit.py            # Bandit environment
│   ├── algorithms/          # Bandit algorithms
│   │   ├── __init__.py
│   │   ├── algorithms.py    # Epsilon-Greedy, UCB, Gradient Bandit, etc.
│   ├── experiment/          # Experiment framework
│   │   ├── __init__.py
│   │   ├── experiment.py
│   ├── plots/               # Visualization functions
│   │   ├── __init__.py
│   │   ├── plots.py
│── main.py                  # Runs experiments & visualizes results
│── requirements.txt          # Dependencies (numpy, matplotlib, etc.)
│── README.md                 # Project documentation
│── .gitignore                # Ignore unnecessary files
```

---

## 📖 Implemented Algorithms

### ✅ **1. Epsilon-Greedy Algorithm**
- Balances exploration and exploitation by selecting a **random action** with probability \( \epsilon \), otherwise choosing the **best-known action**.

### ✅ **2. Upper Confidence Bound (UCB)**
- Uses a confidence bonus to encourage exploration:
  \[
  A_t = \arg\max_a \left[ \hat{Q}_t(a) + c \sqrt{\frac{2 \ln t}{N_t(a)}} \right]
  \]
- Higher \( c \) means **more exploration**.

### ✅ **3. Gradient Bandit Algorithm**
- Uses **softmax action preferences** to **dynamically adapt** to the reward distribution.

---

## 🚀 How to Run Experiments

### 📌 **1. Install Dependencies**
Ensure Python 3+ is installed, then run:
```bash
pip install -r requirements.txt
```

### 📌 **2. Run the Main Experiment Script**
Execute:
```bash
python main.py
```
This runs the **multi-armed bandit simulations**, testing different algorithms **with various parameter values**.

---

## 📊 Parameter Tuning & Visualization

### 📌 **Modify `main.py` to Tune Parameters**
- **Epsilon-Greedy:** Adjust `epsilons = [0, 0.01, 0.1, 0.3, 0.5]`
- **UCB:** Modify `c_values = [0.1, 1, 2, 5]`
- **Gradient Bandit:** Change `alphas = [0.01, 0.1, 0.4]`

---

## 🛠 Future Improvements
✅ Implement **Thompson Sampling**  
✅ Extend to **non-stationary environments**  
✅ Add **grid search for hyperparameter tuning**  

---

## 📜 License
This project is open-source under the **MIT License**.

---

## ⭐️ Acknowledgments
- **Sutton & Barto** for *Reinforcement Learning: An Introduction* 📖  
- **OpenAI, DeepMind, and RL Research Communities** 🧠  

\
