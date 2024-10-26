# CFG Adapter

## Training

On a single GPU:
```bash
accelerate launch --mixed_precision fp16 train_dit_ag.py --data-path /path/to/synthetic_data --results-dir /path/to/results
```
On multi-GPU setup (not tested):
```bash
accelerate launch --multi_gpu --num_processes N --mixed_precision fp16 train_dit_ag.py /path/to/synthetic_data --results-dir /path/to/results
```

## Sampling trajectory data

Only single GPU is supported for now:
```bash
python sample_trajectories.py --output-dir=/path/to/synthetic_data
```

## Evaluating CFG Adapter model

Implemented with torch DDP (will be adapted to `accelerate` soon):
```bash
torchrun --nnodes=1 --nproc_per_node=1 sample_ddp.py --ckpt /path/to/checkpoint.safetensors --sample-dir /path/to/samples --num-fid-samples 50000
```
This can be used to compute FID score and other metrics with [this evaluator script](https://github.com/openai/guided-diffusion/tree/main/evaluations).
