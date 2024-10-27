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

Outputs inception score, FID-10k or FID-50k, precision, recall, sFID-10k, sFID-50k:
```bash
python compute_metrics.py --ckpt /path/to/ckpt_dir --output-dir /path/to/output_dir --ref-batch path/to/VIRTUAL_imagenet256_labeled.npz
```