# Reproducing the benchmark

## Recommended path: automated Nebius H100 run

The project is designed to execute remotely. The workstation only orchestrates the run and receives the completed notebook and artifacts.

### Prerequisites

1. A Nebius project with H100 quota and a subnet that can assign a public IP.
2. An authenticated Nebius CLI profile.
3. Local `bash`, `jq`, `ssh`, `ssh-keyscan`, and `rsync` commands.
4. An Ed25519 SSH key, by default `~/.ssh/id_ed25519` and its `.pub` file.
5. A Hugging Face token with permission to download the selected model.

Verify the key tools:

```bash
nebius version
jq --version
ssh -V
rsync --version
```

### Configure Hugging Face authentication

```bash
cp .env.example .env
```

Edit `.env` and set:

```dotenv
HF_TOKEN=hf_your_token_here
```

`.env` is ignored by Git. Do not place the token in the notebook, README, shell history, or committed configuration.

### Configure Nebius

The automation has assignment-specific defaults. Override them when using another project, subnet, profile, or SSH key:

```bash
export NEBIUS_PROFILE=my-profile
export NEBIUS_PROJECT_ID=my-project-id
export NEBIUS_SUBNET_ID=my-subnet-id
export SSH_PRIVATE_KEY=/absolute/path/to/id_ed25519
```

Optional variables include `SSH_PUBLIC_KEY_PATH`, `SSH_USER`, `REMOTE_DIR`, and `INSTANCE_NAME`.

### Run

```bash
./run-full-project.sh
```

The script performs this lifecycle:

1. Creates a Nebius H100 SXM VM with a 300 GiB managed SSD.
2. Waits for a public IP, SSH, and cloud-init.
3. Uploads the repository, including the ignored `.env` file.
4. Creates a fresh remote `.venv`.
5. Installs `requirements.txt` and runs `pip check`.
6. Verifies CUDA and prints the detected GPU.
7. Executes `quant_serving.ipynb` **in place** with `nbconvert`.
8. Downloads the executed notebook and `results/` directory.
9. Stops the created VM through an exit trap, including on most failures and interruptions.

## Verify completion

After the command succeeds:

```bash
find results -maxdepth 2 -type f | sort
```

Confirm that:

- [`quant_serving.ipynb`](../quant_serving.ipynb) contains outputs for all code cells and no error traceback.
- Both self-check cells print `PASS` and `All checks passed`.
- [`results/q1/q1_results.json`](../results/q1/q1_results.json) and [`snr_vs_bits.png`](../results/q1/snr_vs_bits.png) exist.
- [`results/q2/comparison.json`](../results/q2/comparison.json) contains finite BF16 and FP8 values.
- Four GuideLLM JSON reports and four HTML reports exist under [`results/q2/`](../results/q2/).
- [`bf16_vs_fp8.png`](../results/q2/bf16_vs_fp8.png) exists.

## Inspect reports

- Open the executed notebook for the complete narrative, outputs, and writeup.
- Open `results/q2/bench_*.html` in a browser for detailed GuideLLM reports.
- Use `results/q2/comparison.json` for automation or result comparison.
- Inspect `results/q2/vllm_*.log` when validating model-loading allocations or diagnosing startup.

## Resource cleanup

The automation **stops** its VM; it does not delete the VM or managed disk. A stopped VM can retain billable storage. After preserving the results, delete resources you no longer need from the Nebius console or CLI.

If the script is terminated in a way that prevents its trap from running, verify the instance state manually:

```bash
nebius compute instance get INSTANCE_ID --format json
nebius compute instance stop INSTANCE_ID
```

Never assume cleanup succeeded after a workstation crash, forced process kill, or network outage.
