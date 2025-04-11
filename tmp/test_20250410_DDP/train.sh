# source /home/HPCBase/tools/module-5.2.0/init/profile.sh
# module use /home/HPCBase/modulefiles/
# module purge
# module load compilers/cuda/11.6.0
# module load compilers/gcc/9.3.0
# module load libs/cudnn/8.6.0_cuda11
# module load libs/nccl/2.18.3_cuda11

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module use /home/HPCBase/modulefiles/
module purge
module load compilers/cuda/11.8.0
module load libs/cudnn/8.6.0_cuda11
module load libs/nccl/2.18.3_cuda11
module load compilers/gcc/12.3.0

#!/bin/bash

export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=60
export NCCL_IB_RETRY_CNT=10
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 激活 Python 环境
source /home/share/huadjyin/home/linadi/miniconda3/etc/profile.d/conda.sh
conda activate /home/share/huadjyin/home/linadi/wqr_files/Envs/Env6_Multiomics_20250401/

# 配置 nnodes, node_rank, master_addr
NNODES=$1
HOSTFILE=$2
HOST=`hostname`

flock -x ${HOSTFILE} -c "echo ${HOST} >> ${HOSTFILE}"
MASTER_IP=`head -n 1 ${HOSTFILE}`
echo "Master IP: ${MASTER_IP}"

HOST_RANK=`sed -n "/${HOST}/=" ${HOSTFILE}`
let NODE_RANK=${HOST_RANK}-1

DISTRIBUTED_ARGS="
    --nproc_per_node 4\
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_IP \
    --master_port 30342
 "
echo $DISTRIBUTED_ARGS
echo "
torchrun.sh ---------------
NNODES=${NNODES},
HOST=${HOST},
HOSTFILE=${HOSTFILE},
MASTER_IP=${MASTER_IP},
HOST_RANK=${HOST_RANK},
NODE_RANK=${NODE_RANK}
---------------------------"

# 获取每节点的 GPU 数量
NUM_GPU_PER_NODE=$(nvidia-smi -L | wc -l)

if [ -z "$NUM_GPU_PER_NODE" ] || [ "$NUM_GPU_PER_NODE" -eq 0 ]; then
    echo "No GPUs detected on node $HOST"
    exit 1
fi

# 运行 torchrun
echo "Current Node: ${HOST}"
echo "Number of GPUs per node: ${NUM_GPU_PER_NODE}"
echo "Number of Nodes: ${NNODES}"
echo "Current Node Rank: ${NODE_RANK}"

#torchrun \
#  --nproc_per_node=${NUM_GPU_PER_NODE} --master_port=19920 --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP} \
#    -m /home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer/tmp/test_20250410/test_20250410.py \

torchrun --nproc_per_node=4 --master_port=19930 --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_IP} /home/share/huadjyin/home/linadi/wqr_files/Projs/20250123_Multiomics3/multiminer2/tmp/test_20250410_DDP/test_pretrain.py