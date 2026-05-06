#!/usr/bin/env bash

initials=(0 1 2 3)
batch_sizes=(2 4 6 8 12 16 24 32 48 64 96 128 192 256 384 512 768 1024)
neurons=(2 5 10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400 450 500)
max_iter=200000
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

echo "Start recording training paths for varying learning rate"
for init in "${initials[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for ns in "${neurons[@]}"; do
      python record_path.py --init "$init" --bs 64 --lr "$lr" --neurons "$ns" --num_workers 8 --max_iter "$max_iter"
    done
  done
done

python activity_measure_corr.py
python ccg.py
python compute_dS_energy.py
python compute_entropy.py
python corr_dist.py
python coupling_energy.py
python flux.py
python lk_dist.py
python param_dist.py
python phase_diagram.py
python phase_transition.py
python plot.py
