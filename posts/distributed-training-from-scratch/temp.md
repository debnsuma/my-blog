## Scaling Out: Data Parallelism (DP) Fundamentals

The sequential bottleneck of Gradient Accumulation can be eliminated through parallelization, leading to **Data Parallelism (DP)**.

### 2.1. Parallelizing Micro-Batches: The DP Architecture

In Data Parallelism, data is distributed across $N_d$ GPUs, enabling multiple micro-batches to be processed simultaneously.

- **Replication**: Every GPU maintains a redundant copy of the entire training state: parameters ($\Phi$), gradients ($\nabla \Phi$), and optimizer states ($\Phi_{optim}$)
- **Parallel Computation**: Each GPU performs forward and backward passes independently on its data slice
- **Scaling**: The global batch size scales directly with the number of devices ($N_d$), providing excellent scalability for large batch requirements:
    $$bs = gbs = mbs \cdot grad\_acc \cdot N_d$$

### 2.2. Synchronization: The All-Reduce Primitive

Since each GPU computes a local gradient ($\nabla \Phi_i$), these must be combined (e.g., summed or averaged) to form the global gradient ($\nabla \Phi_{global}$) before the optimization step. This synchronization is achieved using a **collective operation** called **All-Reduce**.

All-Reduce performs two operations: it **reduces** the data (e.g., sums the gradients) across all nodes, then **broadcasts** the resulting collective value back to all participating nodes.

Implementation using the `torch.distributed` API:

```python
#| code-fold: true
#| code-summary: "PyTorch All-Reduce Example (torchrun --nproc_per_node=3 dist_op.py)"

import torch
import torch.distributed as dist

def init_process():
    # Initializes the process group using the efficient nccl backend
    dist.init_process_group(backend='nccl')
    # Assigns a CUDA device based on the node's rank
    torch.cuda.set_device(dist.get_rank())

def example_all_reduce():
    # Setup communication
    init_process()
    
    # Create a tensor whose value is determined by the rank (Rank 0: 1s, Rank 1: 2s, Rank 2: 3s)
    tensor = torch.tensor([dist.get_rank()+1] * 5, dtype=torch.float32).cuda()
    print(f"Before all_reduce on rank {dist.get_rank()}: {tensor}")

    # Perform Summation (1 + 2 + 3 = 6)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # All ranks receive the summed result
    print(f"After all_reduce on rank {dist.get_rank()}: {tensor}")
# Example output on 3 GPUs:
# After all_reduce on rank 0: tensor([6., 6., 6., 6., 6.], device='cuda:0')
# After all_reduce on rank 1: tensor([6., 6., 6., 6., 6.], device='cuda:1')
# After all_reduce on rank 2: tensor([6., 6., 6., 6., 6.], device='cuda:2')
```

This ensures that despite independent computation, all GPUs synchronize on the global gradient before proceeding to the optimization step.

### 2.3. DP Efficiency: Overlapping Communication and Computation

A naive DP pipeline is sequential: Forward Pass $\rightarrow$ Backward Pass (Computation) $\rightarrow$ All-Reduce (Communication) $\rightarrow$ Optimizer Step. This creates idle time where GPUs wait for the All-Reduce to complete.

The solution is to **overlap gradient synchronization with the backward pass**. Since gradient computation for earlier layers completes while later layers are still computing, the All-Reduce for early layers can begin immediately.

1. **Hooks**: This overlap is implemented via **PyTorch hooks**, which trigger the communication (All-Reduce) function for a parameter immediately after its gradient has been computed in the backward pass
2. **Bucketing**: Sending an All-Reduce command for every single parameter is inefficient due to high network overhead with many small packets. Instead, gradients are grouped into larger **"buckets"** and a single, efficient All-Reduce is launched for all gradients within that bucket

### 2.4. The Limitations of Vanilla DP

While highly effective, Data Parallelism cannot scale indefinitely. Throughput benefits diminish severely at ultra-scale (hundreds of GPUs) due to coordination overhead and network saturation.

More fundamentally, DP does not solve the memory problem: it assumes the **entire model static state** ($16\Psi$) **and the activations for one sequence** ($\mathcal{M}_{act}$) **must fit on a single GPU**. For models larger than $\approx 8\text{B}$ parameters, this assumption fails, necessitating memory sharding.

## 3. ZeRO-Redundancy Optimizers (ZeRO): The Sharding Solution

To train models where $16\Psi$ exceeds the memory of a single GPU, the redundancy inherent in Data Parallelism must be eliminated by systematically sharding the static memory components across $N_d$ GPUs. This is the domain of the **ZeRO (Zero Redundancy Optimizer)** family.

### 3.1. ZeRO-1: Sharding Optimizer States ($\Phi_{optim}$)

ZeRO-1 targets the $12\Psi$ bytes of redundant storage represented by the FP32 master weights ($4\Psi$) and the Adam optimizer states ($8\Psi$).

By sharding these components across $N_d$ GPUs, the memory footprint on each device is significantly reduced:

$$\mathcal{M}_{ZeRO-1} = 2\Psi + 2\Psi + \frac{12\Psi}{N_d}$$

- **Operation**: Parameters and gradients (BF16, $4\Psi$) remain replicated for standard forward/backward passes
- **Communication Tax**: The communication cost is $2\Psi$, needed to synchronize gradients and broadcast the necessary parameters/states back to all nodes

### 3.2. ZeRO-2: Sharding Gradients ($\nabla \Phi$)

ZeRO-2 builds on ZeRO-1 by sharding the $2\Psi$ worth of BF16 gradients, eliminating additional redundancy calculated layer-wise.

The memory requirement per GPU is reduced further:

$$\mathcal{M}_{ZeRO-2} = 2\Psi + \frac{2\Psi+12\Psi}{N_d}$$

- **The Reduce-Scatter Primitive**: This sharding is enabled by the **Reduce-Scatter** collective. During the backward pass, instead of an All-Reduce, a Reduce-Scatter operation is used: it **sums** the gradients across all GPUs, then **scatters** only the relevant $1/N_d$ gradient chunk to the GPU that holds the corresponding optimizer state. This ensures that no single GPU needs to store the full $2\Psi$ gradient
- **Communication Tax**: The communication tax for ZeRO-2 remains $2\Psi$ (Reduce-Scatter for gradients, All-Gather for parameters). Since the memory savings are significant with no communication overhead penalty over ZeRO-1, ZeRO-2 is the preferred strategy for large models where parameters still fit on a single GPU

### 3.3. ZeRO-3 / FSDP: Sharding Parameters ($\Phi_{all}$)

**ZeRO-3**, often implemented using **Fully Sharded Data Parallel (FSDP)**, achieves the maximum theoretical memory reduction by sharding all static components: parameters, gradients, and optimizer states.

The memory footprint is reduced to the theoretical minimum:

$$\mathcal{M}_{ZeRO-3} = \frac{16\Psi}{N_d}$$

- **Dynamic Fetching**: Since no single GPU holds the full model, parameters must be dynamically gathered on demand
    1. **Forward Pass**: Just before computing Layer $L$, the parameters are fetched via **All-Gather**. Once computation is done, the parameters are immediately **Flushed** from memory
    2. **Backward Pass**: The process repeats, fetching parameters for gradient computation and performing a **Reduce-Scatter** on the resulting gradients

- **Communication Tax**: This dynamic, layer-wise fetching increases the total communication tax to $3\Psi$ per step. This higher overhead requires aggressive **computation-communication overlap** to ensure the next layer's parameters are pre-fetched while the current layer is still computing, preventing GPU idle time

## 4. The Distributed Playbook: A Comparative Analysis

Choosing the right strategy comes down to a fundamental trade-off: trading memory redundancy for communication complexity. The following table summarizes the key characteristics of these strategies.

| Strategy | Assumptions (Minimum fit on 1 GPU) | Components Parallelized / Sharded | Memory Footprint ($\mathcal{M}$, excl. $\mathcal{M}_{act}$) | Communication Tax ($\mathcal{M}_{comm}$ per step) |
|:---|:---|:---|:---|:---|
| **Vanilla DP** | Full $16\Psi$ state must fit | Batch of samples | $2\Psi+2\Psi+12\Psi$ | $\Psi$ (gradient all-reduce) |
| **ZeRO-1** | All params, all grads, $1/N_d$ optim | Batch + Optimizer States | $2\Psi+2\Psi+\frac{12\Psi}{N_d}$ | $2\Psi$ (reduce-scatter + all-gather) |
| **ZeRO-2** | All params, $1/N_d$ grads, $1/N_d$ optim | Batch + Optimizer + Gradients | $2\Psi+\frac{2\Psi+12\Psi}{N_d}$ | $2\Psi$ |
| **ZeRO-3** | $1/N_d$ params, $1/N_d$ grads, $1/N_d$ optim | Batch + Optimizer + Gradients + Parameters | $\frac{16\Psi}{N_d}$ | $3\Psi$ |

### Strategy Selection Guidelines

- **If $16\Psi$ fits on your GPU**: Use **Optimized Data Parallelism** (with overlap and bucketing) for the lowest communication cost ($\Psi$)
- **If $16\Psi$ is too large but parameters still fit** (e.g., up to $\approx 70\text{B}$): Use **ZeRO-2**. It provides optimal balance, offering substantial memory savings (by sharding gradients and optimizer states) for a moderate $2\Psi$ communication cost
- **If parameters themselves must be sharded** (e.g., $>100\text{B}$): Use **ZeRO-3/FSDP**. Accept the $3\Psi$ communication cost for maximum memory scaling, constrained only by the aggregate memory of your entire GPU cluster

## 5. Summary and Next Steps

The distributed training playbook provides a set of carefully engineered solutions that systematically address the two major bottlenecks in training LLMs: high activation memory and high static model memory.

This guide covered basic memory management techniques (Selective Checkpointing and Gradient Accumulation) and progressed to horizontal scaling (Data Parallelism). The breakthrough came with the ZeRO progression, which transformed the problem of fitting the model from an individual GPU constraint to a **cluster-wide aggregation problem**.

However, even with ZeRO-3, a fundamental limitation remains: the memory required for the activations of a **single sequence** must still fit on one GPU. As sequence lengths grow for long-context understanding (e.g., $>8192$), future research will increasingly focus on orthogonal techniques like Mixture of Experts (MoE) architectures and highly efficient kernels like Flash Attention to continue pushing the boundaries of distributed training.