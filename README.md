# WCCV_Style-Composition-within-Distinct-LoRA-modules-for-Traditional-Art


## 1. Train each style using LoRA adapters

```
CUDA_VISIBLE_DEVICES=5 accelerate launch /workspace/train_dreambooth_lora_flux_schnell.py \
    --instance_data_dir '/data/trad_art_25/style_data/color_painting/gongpil' \
    --output_dir "/data/trad_art_25/flux/outputs/color_painting/gongpil/" \
    --proj_name "gongpil" \
    --instance_prompt "zzv style" \
    --mixed_precision "bf16" \
    --resolution 512 \
    --train_batch_size 1 \
    --guidance_scale 1 \
    --optimizer "prodigy" \
    --learning_rate 1.0 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --max_train_steps 3500 \
    --seed 0 \
    --rank 16 \
    --validation_epochs 3500​
```

```
CUDA_VISIBLE_DEVICES=5 accelerate launch /workspace/train_dreambooth_lora_flux_schnell.py \
    --instance_data_dir '/data/trad_art_25/style_data/color_painting/ilpil' \
    --output_dir "/data/trad_art_25/flux/outputs/color_painting/ilpil/" \
    --proj_name "ilpil" \
    --instance_prompt "zzv style" \
    --mixed_precision "bf16" \
    --resolution 512 \
    --train_batch_size 1 \
    --guidance_scale 1 \
    --optimizer "prodigy" \
    --learning_rate 1.0 \
    --lr_scheduler "constant" \
    --lr_warmup_steps 0 \
    --max_train_steps 3500 \
    --seed 0 \
    --rank 16 \
    --validation_epochs 3500
```

## 2. Style composition of distinct LoRA models
Put prompts in the style_mixing.py file, then run

```
CUDA_VISIBLE_DEVICES=5 python3 /workspace/style_mixing.py \
    --output_path '/samples' \
    --project_name '/gongpil_ilpil' \
    --lora_path1 '/data/trad_art_25/flux/outputs/color_painting/gongpil/checkpoint-3500' \
    --lora_path2 '/data/trad_art_25/flux/outputs/color_painting/ilpil/checkpoint-3500' \
```
