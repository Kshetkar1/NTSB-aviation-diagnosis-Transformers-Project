# 🔬 Analyze Command

Use this command to interpret code, data, results, or notebooks with deep context awareness.

## What This Does

The "Insight Engine" of your workflow. It adapts its behavior based on what you point it at.
- **Code:** Explains logic, complexity, and mathematical correctness.
- **Results:** Interprets plots, metrics, and logs (RL focus).
- **Notebooks:** Reviews methodology and statistical validity.

## How to Use

Type `/analyze` followed by your target:

**Examples:**
- "Analyze `results/experiment_1/` - is the agent converging?"
- "Analyze this notebook: `notebooks/data_clean.ipynb`. Are there data leaks?"
- "Analyze the math in `rewards.py`. Does it match the PPO objective?"
- "Analyze the chat history. What were our key hypotheses?"

## Modes

**1. Data Scientist Mode (Results/Plots)**
- Correlates `metrics.csv` with `plots.png`.
- Checks for: Variance, Collapsing, Overfitting.
- Output: "The agent learns until episode 100, then collapses. Likely due to high learning rate (see `config.yaml`)."

**2. Peer Review Mode (Notebooks)**
- Checks: Cell execution order, variable shadowing, hidden state.
- Output: "Warning: You define `df` in cell 3 but overwrite it in cell 10 without reloading."

**3. Math Review Mode (Code)**
- Maps implementation to LaTeX equations.
- Output: "Your implementation of Kullback-Leibler divergence is missing the log-sum-exp trick for stability."

## Tips

- Be specific about *what* to look for (e.g., "Analyze for stability").
- Provide context: "This run used hyperparameters X and Y."

---

**Ready?** Point me to what needs analysis.
