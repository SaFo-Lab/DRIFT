# DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents

[Hao Li](https://leolee99.github.io/), [Xiaogeng Liu](https://sheltonliu-n.github.io/), [Hung-Chun Chiu](https://qhjchc.notion.site/), [Dianqi Li](https://scholar.google.com/citations?user=K40nbiQAAAAJ&hl=en), [Ning Zhang](https://cybersecurity.seas.wustl.edu/index.html), [Chaowei Xiao](https://xiaocw11.github.io/).

<p align="center" width="80%">
<a target="_blank"><img src="assets/framework.png" alt="framework" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

The official implementation of NeurIPS 2025 paper "[DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](https://www.arxiv.org/pdf/2506.12104)".

## How to Start on ASB
We provide the evaluation of DRIFT on GPT-4o-mini and GPT-4o on ASB, you can reproduce the results following:

### Construct Your Environment
We utilize the same environment setup as the ASB repository; please refer to the full version in `ASB_README.md` to set up your environment.

```bash
conda create -n ASB python=3.11
source activate ASB
cd ASB_DRIFT
```
You can install the dependencies using
```bash
pip install -r requirements.txt
```


### Set Your OPENAI API KEY
```bash
export OPENAI_API_KEY=your_key
```

### Evaluation

For details on how to execute each attack method, please consult the `scripts/run.sh` file. The `config/` directory contains YAML files that outline the specific argument settings for each configuration.

```python
python scripts/agent_attack.py --cfg_path config/clean.yml # No Attack
python scripts/agent_attack.py --cfg_path config/DPI.yml # Direct Prompt Injection
python scripts/agent_attack.py --cfg_path config/OPI.yml # Observation Prompt Injection
python scripts/agent_attack.py --cfg_path config/MP.yml # Memory Poisoning attack
python scripts/agent_attack.py --cfg_path config/mixed.yml # Mixed attack
python scripts/agent_attack_pot.py # PoT backdoor attack
```

In our paper, we report results for Observation Prompt Injection (OPI).