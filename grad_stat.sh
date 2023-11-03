#!/bin/zsh

# 声明一个字符串数组，用于存储不同的transfer参数
transfer_params=()

# 使用for循环迭代数组中的每一个元素
for transfer_param in "${transfer_params[@]}"; do
  # 执行torchrun命令，使用当前迭代的transfer参数
  torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --batch-gpu=16 --transfer="$transfer_param"
done

