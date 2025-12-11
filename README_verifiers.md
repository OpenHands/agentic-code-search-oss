# Instructions for using the verifiers environment

1. Install dependencies

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

2. Clone some repos from the SWE-bench dataset

```bash
uv run scripts/clone_repos.py --output-dir ./swebench_repos --dataset princeton-nlp/SWE-bench_Lite --max-workers 10
```

3. Run `vllm` and serve `Qwen3-8B`
```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser deepseek_r1
```

4. Install [ripgrep](https://github.com/BurntSushi/ripgrep?tab=readme-ov-file#installation)
```bash
sudo apt-get install ripgrep -y
```

5. Run the verifiers eval with your model of choice

```bash
uv run vf-eval swe-grep-oss-env --api-base-url http://localhost:8000/v1 --model "Qwen/Qwen3-8B" --num-examples 1 --rollouts-per-example 1
```
