#!/usr/bin/env bash

initials=(0)
batch_sizes=(64)
neurons=(50)
max_iter=200
learning_rates=()
# learning rate adjust to have the same covariance as batch size
learning_rates=($(python3 -c "import sys; print(' '.join(f'{0.1*64/float(bs):.8f}' for bs in sys.argv[1:]))" "${batch_sizes[@]}"))

echo initials: "${initials[@]}"
echo learning rates: "${learning_rates[@]}"
echo batch sizes: "${batch_sizes[@]}"
echo neurons: "${neurons[@]}"

echo "Start recording training paths for varying batch size"
for init in "${initials[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    for ns in "${neurons[@]}"; do
      python record_path.py --init "$init" --bs "$bs" --lr 0.1 --neurons "$ns" --num_workers 8 --max_iter "$max_iter" --overwrite
    done
  done
done
