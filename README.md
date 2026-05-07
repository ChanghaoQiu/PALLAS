# PALLAS
PALLAS is a **P**er-slot **A**djustment and **L**oad-**L**earned **A**daptive **S**haping defense for multi-tab WF.
PALLAS consists of three key modules that together realize template-driven shaping framework.

## Module 1 — State-Template Learning

Learns traffic states and per-state template codebooks from training data.

```bash
python state_template_learning.py \
    --train_npz ./data/train.npz \
    --out       ./config/stl_config.pkl \
    --delta 0.2 --K_slot 30 --num_states 2 \
    --max_templates 48 --max_global_templates 64
```

## Module 2 — Template Smoothing

Blurs codebook boundaries between adjacent states.

```bash
python template_smoothing.py \
    --config ./config/stl_config.pkl \
    --out    ./config/smoothed_config.pkl \
    --alpha 0.4 --sigma 1.3
```

## Module 3 — Online Regularization

Applies the shaping policy to dataset splits.

```bash
python online_regularization.py \
    --config   ./config/smoothed_config.pkl \
    --data_dir ./data/closed_world \
    --out_dir  ./defended \
    --tradeoff 0.05 --drain_X 4 --idle_slots_no_real 2
```
