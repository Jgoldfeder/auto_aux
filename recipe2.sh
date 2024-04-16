#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python3 train.py torch/dogs --dataset torch/dogs  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "dogs_new" --name "seed 5 resnet50 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet50 --num-classes 120  --log-wandb  --dataset-download


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/dogs --dataset torch/dogs  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "dogs_new" --name "seed 5   resnet50 " --num-classes 120 --seed 5 --model=resnet50 --dataset-download  --log-wandb  


CUDA_VISIBLE_DEVICES=1 python3 train.py torch/dogs --dataset torch/dogs  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "dogs_new" --name "seed 5   resnet101 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet101 --num-classes 120  --log-wandb  


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/dogs --dataset torch/dogs  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "dogs_new" --name "seed 5   resnet101 " --num-classes 120 --seed 5 --model=resnet101 --dataset-download  --log-wandb 



CUDA_VISIBLE_DEVICES=1 python3 train.py torch/dogs --dataset torch/dogs  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "dogs_new" --name "seed 5   vit_base_patch16_224_miil_in21k dual 1 1" --dual --weights 1 1 --seed 5 --model=vit_base_patch16_224_miil_in21k --num-classes 120  --log-wandb  


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/dogs --dataset torch/dogs  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "dogs_new" --name "seed 5   vit_base_patch16_224_miil_in21k " --num-classes 120 --seed 5 --model=vit_base_patch16_224_miil_in21k --dataset-download  --log-wand















CUDA_VISIBLE_DEVICES=1 python3 train.py torch/cub2011 --dataset torch/cub2011  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "cub2011_new" --name "seed 5 resnet50 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet50 --num-classes 200  --log-wandb  --dataset-download


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/cub2011 --dataset torch/cub2011  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "cub2011_new" --name "seed 5   resnet50 " --num-classes 200 --seed 5 --model=resnet50 --dataset-download  --log-wandb  


CUDA_VISIBLE_DEVICES=1 python3 train.py torch/cub2011 --dataset torch/cub2011  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "cub2011_new" --name "seed 5   resnet101 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet101 --num-classes 200  --log-wandb  


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/cub2011 --dataset torch/cub2011  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "cub2011_new" --name "seed 5   resnet101 " --num-classes 200 --seed 5 --model=resnet101 --dataset-download  --log-wandb 



CUDA_VISIBLE_DEVICES=1 python3 train.py torch/cub2011 --dataset torch/cub2011  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "cub2011_new" --name "seed 5   vit_base_patch16_224_miil_in21k dual 1 1" --dual --weights 1 1 --seed 5 --model=vit_base_patch16_224_miil_in21k --num-classes 200  --log-wandb  


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/cub2011 --dataset torch/cub2011  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "cub2011_new" --name "seed 5   vit_base_patch16_224_miil_in21k " --num-classes 200 --seed 5 --model=vit_base_patch16_224_miil_in21k --dataset-download  --log-wandb






























CUDA_VISIBLE_DEVICES=1 python3 train.py torch/caltech256 --dataset torch/caltech256  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "caltech256_new" --name "seed 5 resnet50 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet50 --num-classes 257  --log-wandb  


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/caltech256 --dataset torch/caltech256  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "caltech256_new" --name "seed 5   resnet50 " --num-classes 257 --seed 5 --model=resnet50 --dataset-download  --log-wandb  


CUDA_VISIBLE_DEVICES=1 python3 train.py torch/caltech256 --dataset torch/caltech256  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "caltech256_new" --name "seed 5   resnet101 dual 1 1" --dual --weights 1 1 --seed 5 --model=resnet101 --num-classes 257  --log-wandb  


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/caltech256 --dataset torch/caltech256  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "caltech256_new" --name "seed 5   resnet101 " --num-classes 257 --seed 5 --model=resnet101 --dataset-download  --log-wandb 



CUDA_VISIBLE_DEVICES=1 python3 train.py torch/caltech256 --dataset torch/caltech256  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "caltech256_new" --name "seed 5   vit_base_patch16_224_miil_in21k dual 1 1" --dual --weights 1 1 --seed 5 --model=vit_base_patch16_224_miil_in21k --num-classes 257  --log-wandb  


CUDA_VISIBLE_DEVICES=1  python3 train.py torch/caltech256 --dataset torch/caltech256  --val-split eval -b=64  --epochs=50 --color-jitter=0  --sched='cosine' --model-ema --model-ema-decay=0.995 --reprob=0.5 --smoothing=0.1 --min-lr=1e-8 --warmup-epochs=3 --train-interpolation=bilinear --aa=v0 --pretrained --lr=2e-4  --opt=adam --weight-decay=1e-4 --experiment "caltech256_new" --name "seed 5   vit_base_patch16_224_miil_in21k " --num-classes 257 --seed 5 --model=vit_base_patch16_224_miil_in21k --dataset-download  --log-wandb