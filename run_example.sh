torchrun --nnodes=1 \
    --max_restarts=0  \
    --rdzv_id=${RANDOM} \
    --rdzv_backend=c10d \
    --nproc_per_node=gpu  \
    --rdzv_endpoint=localhost:29403  -m \
    examples.train_single_gpu