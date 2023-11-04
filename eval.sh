#!/usr/bin/env zsh
for file_name in `ls training-runs/switch-back_g16/*pkl`
do
  # echo snapshot name
  echo ${file_name}

  # calculate training gradient statistics
  torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs --data=datasets/cifar10-32x32.zip \
    --cond=1 --arch=ddpmpp --batch-gpu=2 --transfer=${file_name}

  # clean up fid-tmp folder
  rm fid-tmp -rf

  # generate 50000 images and save them as fid-tmp/*/*.png
  torchrun --standalone --nproc_per_node=8 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
    --network=${file_name}

  # calculate FID
  torchrun --standalone --nproc_per_node=8 fid.py calc --images=fid-tmp \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --pkl=${file_name} --out_dir='.' --file_n='fid_result_fqt_switchback.txt'
done