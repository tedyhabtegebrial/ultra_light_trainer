srun \
-K \
--partition=RTX3090 \
--gpus 2 --ntasks 2 --nodes 1 \
--time 00:30:00 \
--container-image=/netscratch/teddy/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
--container-workdir="`pwd`"  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,`pwd`:`pwd` \
--mem-per-cpu=16G torchrun \
--nnodes=1 \
--max_restarts=0  \
--rdzv_id=${RANDOM} \
--rdzv_backend=c10d \
--nproc_per_node=gpu  \
--rdzv_endpoint=localhost:29403  -m examples.train_mul_gpu


srun \
--partition=RTX3090 \
--gpus 2 --ntasks 2 --nodes 1 \
--time 00:30:00 \
--container-image=/netscratch/teddy/enroot/nvcr.io_nvidia_pytorch_23.08-py3.sqsh \
--container-workdir="`pwd`"  --container-mounts=/netscratch:/netscratch,/ds:/ds:ro,`pwd`:`pwd` \
--mem-per-cpu=16G --pty /bin/bash