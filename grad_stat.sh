#!/bin/zsh

# 声明一个字符串数组，用于存储不同的transfer参数
transfer_params=(
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g8/network-snapshot-005018.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g8/network-snapshot-007526.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g8/network-snapshot-010035.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g8/network-snapshot-012544.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g16/network-snapshot-000000.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g16/network-snapshot-002509.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g16/network-snapshot-005018.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g16/network-snapshot-007526.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g17/network-snapshot-002509.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g17/network-snapshot-007526.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision-g17/network-snapshot-010035.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision/network-snapshot-000000.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision/network-snapshot-002509.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision/network-snapshot-005018.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision/network-snapshot-007526.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision/network-snapshot-010035.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision/network-snapshot-012544.pkl"
  "/home/yezihao-fwxz/edm/training-runs/full-precision/network-snapshot-045158.pkl"
)

# 使用for循环迭代数组中的每一个元素
for transfer_param in "${transfer_params[@]}"; do
  # 执行torchrun命令，使用当前迭代的transfer参数
  torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp --batch-gpu=16 --transfer="$transfer_param"
done

