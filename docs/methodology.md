# Benchmark methodology

## Objective

The project tests two connected questions:

1. How do bit width and scale granularity affect reconstruction error and theoretical weight memory?
2. How does serving the same model in BF16 and on-the-fly FP8 affect measured GPU memory, throughput, and latency?

## Hardware and software

The recorded Q2 run used:

- NVIDIA H100 80 GB HBM3, compute capability 9.0.
- Nebius `gpu-h100-sxm`, preset `1gpu-16vcpu-200gb`.
- Ubuntu 24.04 CUDA 13 image.
- Python 3.12, PyTorch 2.11, vLLM 0.25.1, and GuideLLM 0.7.1.
- `Qwen/Qwen2.5-7B-Instruct`, served from the same Hugging Face model ID for both configurations.

The exact dependency declarations are in [`requirements.txt`](../requirements.txt). Server logs and GuideLLM reports preserve additional run evidence under [`results/q2/`](../results/q2/).

## Q1: quantization experiment

Q1 uses a deterministic 256×1024 synthetic weight matrix containing four outlier columns. It compares:

- Symmetric and asymmetric integer mappings.
- Per-tensor and per-channel scale granularity.
- 8-, 6-, 4-, and 3-bit representations.

Reconstruction quality is measured with:

- Mean squared error (MSE).
- Maximum absolute error.
- Signal-to-noise ratio: `10 × log10(mean(w²) / MSE)`.

The 7B memory table is a weight-only theoretical estimate: `parameters × bits / 8`, converted to GiB. It excludes scales, metadata, activations, KV cache, allocator overhead, and runtime workspaces.

## Q2: serving experiment

The independent variable is the vLLM quantization mode:

- **BF16:** the base model is loaded without a quantization flag.
- **FP8:** the same base model is loaded with `--quantization fp8`.

Both configurations use:

- Maximum model length: 4096 tokens.
- GPU memory utilization target: 0.85.
- Synthetic requests: 512 prompt tokens and 256 output tokens.
- Benchmark duration: 30 seconds per profile.
- Synchronous profile for single-stream latency.
- Throughput profile with maximum concurrency 64 for saturation throughput.

The server is stopped between configurations so BF16 and FP8 do not share GPU allocations.

## Metrics

| Metric | Meaning |
|---|---|
| Weight allocation | Model-loading GPU allocation reported by vLLM |
| TTFT | Time from request submission to the first generated token |
| ITL | Delay between consecutive generated tokens |
| Request latency | End-to-end request duration |
| Output tokens/s | Aggregate token generation throughput |
| Requests/s | Completed request throughput |

Weight allocation is parsed from vLLM logs rather than `nvidia-smi`: vLLM also reserves memory for the KV cache and runtime, so total process memory is not an isolated weight measurement.

## Fairness controls

- The same GPU, model ID, context limit, request shape, benchmark duration, and memory-utilization target are used for BF16 and FP8.
- The model is downloaded before timed requests begin.
- vLLM health is checked before GuideLLM starts.
- Every benchmark emits JSON and HTML reports for independent inspection.
- The notebook keeps all official cell outputs.

## Interpretation limits

- Results describe this model, software version, request shape, and H100 configuration; they are not universal performance guarantees.
- A single benchmark run does not quantify run-to-run variance or confidence intervals.
- Synthetic traffic is controlled but may not represent a production prompt distribution.
- The serving benchmark measures performance, not model quality. FP8 should also be evaluated on representative accuracy tasks before deployment.
- TTFT and ITL respond differently because prefill and autoregressive decode have different compute and memory behavior.

## Recorded result

In the committed run, FP8 reduced model-loading allocation from 14.29 GiB to 8.20 GiB, increased peak output throughput from approximately 5,836 to 8,068 tokens/s, and reduced median ITL from 6.03 ms to 4.11 ms. Full precision values are stored in [`comparison.json`](../results/q2/comparison.json).
