# Troubleshooting

## Nebius CLI cannot connect or authenticate

Symptoms include DNS errors, authentication timeouts, or permission-denied responses.

1. Confirm normal internet and DNS access.
2. Verify the selected profile: `nebius profile list`.
3. Re-authenticate the profile if its credentials expired.
4. Confirm that `NEBIUS_PROJECT_ID` and `NEBIUS_SUBNET_ID` belong to the same accessible environment.

## H100 VM creation fails

Common causes are quota, regional capacity, an invalid subnet, or insufficient project permissions.

- Check H100 quota and availability in the selected region.
- Confirm the configured platform and preset are available.
- Confirm that the subnet permits public IP allocation.
- Use a unique `INSTANCE_NAME` when a naming conflict exists.

The automation stops only instances for which it successfully captured an instance ID. If creation partially succeeds, check the Nebius console for leftover resources.

## SSH never becomes ready

- Allow cloud-init several minutes on a new CUDA image.
- Confirm the public key matches `SSH_PRIVATE_KEY`.
- Verify port 22 is reachable from the workstation.
- Confirm the VM received a public IP.
- Remove no host keys manually: the script uses a run-specific temporary `known_hosts` file.

## Package installation fails

Use a fresh virtual environment. vLLM selects a compatible PyTorch/CUDA stack, and reusing an unrelated environment can create binary conflicts.

On the VM:

```bash
cd /home/sapir/Hometask_3_Quantization_and_Benchmarking
rm -rf .venv
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

If compilation reports a missing `Python.h`, install `python3-dev`. If it cannot find a compiler or `ninja`, ensure `build-essential` is installed and the virtual environment is activated when launching vLLM.

## PyTorch cannot see the GPU

Run:

```bash
nvidia-smi
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
```

For this project, the H100 should report compute capability `(9, 0)`. Driver/runtime incompatibility usually requires using a matching Nebius CUDA image or rebuilding the virtual environment.

## Hugging Face warns about unauthenticated requests

Confirm `.env` exists in the project root and contains `HF_TOKEN`. The full-run script exports `.env` before notebook execution. Do not print or commit the value.

If a token has been pasted into chat, logs, or a public issue, revoke it and issue a new token even if Git never tracked it.

## vLLM exits during startup

Inspect:

```bash
tail -n 100 results/q2/vllm_bf16.log
tail -n 100 results/q2/vllm_fp8.log
```

Typical causes include:

- Unsupported model architecture or quantization method.
- Gated model access without appropriate Hugging Face permissions.
- Insufficient GPU memory.
- CUDA/PyTorch binary mismatch.
- A missing executable because vLLM was started without the virtual environment on `PATH`.

Smoke-test the selected H100 model with:

```bash
source .venv/bin/activate
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --quantization fp8 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85
```

Stop the smoke test before running the notebook.

## GuideLLM produces no report

- Check `http://127.0.0.1:8000/health` while vLLM is running.
- Inspect the vLLM log for an early server exit.
- Confirm GuideLLM 0.7.1 is installed; its CLI schema differs from older releases.
- Ensure the output directory is writable.
- Increase the subprocess timeout only after confirming requests are progressing.

## Weight memory is `NaN`

vLLM log wording has changed across versions. The notebook supports both legacy `weights took X GiB` and current `Model loading took X GiB` messages. If a future version changes this again, inspect `results/q2/vllm_*.log` and update the compatibility parser without changing the meaning of the measurement.

Do not substitute total `nvidia-smi` process memory: it includes KV-cache reservation and runtime allocations.

## Notebook page shows old answers or results

The remote Jupyter browser can retain an already-open document model. Refresh the browser tab or close and reopen `quant_serving.ipynb` after an automated run downloads or replaces the file.

There should be one notebook only. The automation executes `quant_serving.ipynb` in place and does not create `.executed.ipynb` or `.rerun.ipynb` copies.

## VM did not stop

The script uses an exit trap, but `kill -9`, a host crash, or a prolonged network failure can prevent cleanup. Locate and stop the instance manually in Nebius. After preserving results, delete the stopped VM and managed disk if they are no longer needed.
