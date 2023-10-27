#!/usr/bin/env zsh

start_point=-1


for file_name in `ls training-runs/full-precision-g17/*pkl`
do
  number=${file_name##*-}
  number=${number%%.pkl}
  if (( number > start_point )); then
    echo ${file_name}


    rm fid-tmp -rf

    # Generate 50000 images and save them as fid-tmp/*/*.png
    torchrun --standalone --nproc_per_node=7 generate.py --outdir=fid-tmp --seeds=0-49999 --subdirs \
      --network=${file_name}


    # Calculate FID
    torchrun --standalone --nproc_per_node=7 fid.py calc --images=fid-tmp \
      --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz --pkl=${file_name} --out_dir='.' --file_n='fid_result_fqt_switchback.txt'
  fi
done