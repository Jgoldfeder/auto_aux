#!/bin/bash




CUDA_VISIBLE_DEVICES=2 python3 train.py torch/aircraft --dataset torch/aircraft  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "aircraft_new" --name "seed 5  resnet50 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet50 --num-classes 102  --log-wandb 


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/aircraft --dataset torch/aircraft  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "aircraft_new" --name "seed 5  resnet50 " --num-classes 102 --seed 5 --model=resnet50 --dataset-download  --log-wandb  



CUDA_VISIBLE_DEVICES=2 python3 train.py torch/aircraft --dataset torch/aircraft  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "aircraft_new" --name "seed 5  resnet101 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet101 --num-classes 102  --log-wandb  


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/aircraft --dataset torch/aircraft  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "aircraft_new" --name "seed 5  resnet101 " --num-classes 102 --seed 5 --model=resnet101 --dataset-download  --log-wandb 



CUDA_VISIBLE_DEVICES=2 python3 train.py torch/aircraft --dataset torch/aircraft  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "aircraft_new" --name "seed 5  vit_base_patch16_224_miil_in21k dual 1 1" --dual --weights 1 1 --seed 5 --model=vit_base_patch16_224_miil_in21k --num-classes 102  --log-wandb  


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/aircraft --dataset torch/aircraft  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "aircraft_new" --name "seed 5  vit_base_patch16_224_miil_in21k " --num-classes 102 --seed 5 --model=vit_base_patch16_224_miil_in21k --dataset-download  --log-wandb













CUDA_VISIBLE_DEVICES=2 python3 train.py torch/flowers102 --dataset torch/flowers102  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "flowers102_new" --name "seed 5 resnet50 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet50 --num-classes 102  --log-wandb  --dataset-download


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/flowers102 --dataset torch/flowers102  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "flowers102_new" --name "seed 5   resnet50 " --num-classes 102 --seed 5 --model=resnet50 --dataset-download  --log-wandb  



CUDA_VISIBLE_DEVICES=2 python3 train.py torch/flowers102 --dataset torch/flowers102  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "flowers102_new" --name "seed 5   resnet101 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet101 --num-classes 102  --log-wandb  


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/flowers102 --dataset torch/flowers102  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "flowers102_new" --name "seed 5   resnet101 " --num-classes 102 --seed 5 --model=resnet101 --dataset-download  --log-wandb 



CUDA_VISIBLE_DEVICES=2 python3 train.py torch/flowers102 --dataset torch/flowers102  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "flowers102_new" --name "seed 5   vit_base_patch16_224_miil_in21k dual 1 1" --dual --weights 1 1 --seed 5 --model=vit_base_patch16_224_miil_in21k --num-classes 102  --log-wandb  


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/flowers102 --dataset torch/flowers102  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "flowers102_new" --name "seed 5   vit_base_patch16_224_miil_in21k " --num-classes 102 --seed 5 --model=vit_base_patch16_224_miil_in21k --dataset-download  --log-wandb




















CUDA_VISIBLE_DEVICES=2 python3 train.py torch/food101 --dataset torch/food101  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "food101_new" --name "seed 5 resnet50 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet50 --num-classes 101  --log-wandb  --dataset-download


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/food101 --dataset torch/food101  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "food101_new" --name "seed 5   resnet50 " --num-classes 101 --seed 5 --model=resnet50 --dataset-download  --log-wandb  



CUDA_VISIBLE_DEVICES=2 python3 train.py torch/food101 --dataset torch/food101  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "food101_new" --name "seed 5   resnet101 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet101 --num-classes 101  --log-wandb  


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/food101 --dataset torch/food101  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "food101_new" --name "seed 5   resnet101 " --num-classes 101 --seed 5 --model=resnet101 --dataset-download  --log-wandb 



CUDA_VISIBLE_DEVICES=2 python3 train.py torch/food101 --dataset torch/food101  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "food101_new" --name "seed 5   vit_base_patch16_224_miil_in21k dual 1 1" --dual --weights 1 1 --seed 5 --model=vit_base_patch16_224_miil_in21k --num-classes 101  --log-wandb  


CUDA_VISIBLE_DEVICES=2  python3 train.py torch/food101 --dataset torch/food101  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "food101_new" --name "seed 5   vit_base_patch16_224_miil_in21k " --num-classes 101 --seed 5 --model=vit_base_patch16_224_miil_in21k --dataset-download  --log-wandb



