
# 🤖🧠 Embodied Agent Interface (EAI)
**Benchmarking LLMs for Embodied Reasoning and Decision Making with Safety Constraints**


🔗 Original repository:  
https://github.com/embodied-agent-interface/embodied-agent-interface

---

## 📌 Overview

This benchmark provides a **comprehensive assessment** of LLM performance across different subtasks, identifying their **strengths and weaknesses** in embodied decision-making contexts with **safety constraints**.

It includes:

1. ✅ Tasks that can be achieved **safely** under **adversarial** and **multi-constraint** settings  
2. ⚠️ **Adversarial instructions** that the agent must **avoid**

---

## 🛠️ Running the Experiments

We consider two planning strategies:

- 🚀 **One-go Planning** → ideal for **direct planning**
- 🔁 **Stepwise Planning** → ideal for **iterative planning**

---


## 🚀🧠 One-go Planning

In the **one-go** planning approach, the agent 🤖 generates a plan 📋 (a sequence of actions) in a **single attempt** for a given instruction or task.  
▶️ The plan is then executed by the simulator 🎮.  
📊 Upon successful execution, the environment graph 🗺️ **𝒢\*** (representing the updated state of the environment) is evaluated.

💡 Thanks to its straightforward design, this planning strategy is well-suited for **direct planning** scenarios.

### ▶️ Example Command

```bash
python src/behavior_eval/evaluation/action_sequencing/scripts/evaluate_results.py \
    --mode onego \
    --llm_name LGAI-EXAONE/EXAONE-3.5-32B-Instruct \
    --strategy direct
```

---


## 🔁 Stepwise Planning

In **stepwise planning**, the agent 🤖 interacts with the environment 🌍 for *n* steps and *m* trials to finish the task.  
At each step, the agent selects and executes an action \( a_{ij} \), where \( i \in m \) and \( j \in n \), which is processed by the simulator 🎮 and returns an observation \( o_{ij} \), along with the updated environment state \( \mathcal{G}_{ij} \).

This interaction continues for a (pre-defined) number of steps, constituting a trajectory  
\( \tau_i = \{ a_{11}, o_{11}, a_{12}, o_{12}, \ldots \} \).  

At the end of each trial, a critic 🧠 evaluates the (so-far generated) plan \( \mathcal{P}_i \) and provides feedback \( f_i \).  
This feedback guides the agent in refining its strategy or decision-making process for future trials.

The process continues until the agent generates the **“Done”** action or exhausts all trials.  
If the plan is executable, the updated environment state \( \mathcal{G}^\* \) is passed on for evaluation.

✨ This strategy suits **iterative planning**, i.e., where interactions occur between the agent and the simulator/environment.

⚙️ Key Arguments

```bash
`--llm_name`: `Qwen/Qwen3-32B`  `gpt-4.1-mini` etc 
`--strategy`: `direct`  `react` `rej`
`--rm_safety_instruction`: if set, then instruction for safety enhancement in the prompt will be removed, for ablation study.
`--reflex`: if set, `reflexion` or `AI-critic from gpt-4.1` will be applied.
`--reflex_from_llm_as_judge`: if set, use `AI-critic from gpt-4.1`, otherwise, use `reflexion` 
`--trial`: number of rounds for `reflexion` or `AI-critic`
```

🧪 Example Experiments

🔁 Stepwise + ReAct

```bash
python src/behavior_eval/evaluation/action_sequencing/scripts/evaluate_results.py \
--mode stepwise --llm_name ${llm}  --strategy react 
```

🧠 Stepwise + ReAct + Reflexion / AI-Critic

```bash
# reflexion for additional N rounds (N is 1,2,3,...)
python src/behavior_eval/evaluation/action_sequencing/scripts/evaluate_results.py \
--mode stepwise --llm_name ${llm}  --strategy react \
--reflex_from_llm_as_judge \
--trial 1 
```

🧹 Ablation: Remove Safety Instruction

```bash
python src/behavior_eval/evaluation/action_sequencing/scripts/evaluate_results.py \
--mode onego --strategy direct --llm_name ${llm}    \
--rm_safety_instruction

python src/behavior_eval/evaluation/action_sequencing/scripts/evaluate_results.py \
--mode stepwise --strategy direct --llm_name ${llm}    \
--rm_safety_instruction 
```

⚠️ Risk-Specific Evaluation (e.g., Electrical Hazard)

```bash
python src/behavior_eval/evaluation/action_sequencing/scripts/evaluate_results.py \
--mode stepwise --strategy react --llm_name ${llm}   \
--reflex_from_llm_as_judge \
--trial 1 \
--risk_category 'electrical hazard' 

```



<!-- # BibTex

If you find our work helpful, please consider citing it:

```bash
@inproceedings{sadhu2025vestabench,
  title={VESTABENCH: An Embodied Benchmark for Safe Long-Horizon Planning Under Multi-Constraint and Adversarial Settings},
  author={Sadhu, Tanmana and Chen, Yanan and Pesaranghader, Ali},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track},
  year={2025}
}
``` -->