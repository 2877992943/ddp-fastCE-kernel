"""https://pytorch.org/tutorials/intermediate/ddp_tutorial.html


python -c "from os import path; import torch; print(path.join(path.dirname(torch.__file__), 'distributed', 'launch.py'))"

python /path/to/launch.py --nnode=1 --node_rank=0 --nproc_per_node=2 example.py --local_world_size=2

"""
import os
import time
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
torch.manual_seed(42)
import argparse

from torch.nn.parallel import DistributedDataParallel as DDP
from cross_entropy_loss_see1 import fast_cross_entropy_loss

import torch.nn.functional as nn
ce=nn.cross_entropy


bsz,seql=100,8

#nnflag=True
nnflag=False

class ToyModel(torch.nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        # self.net1 = nn.Linear(10, 10)
        # self.relu = nn.ReLU()
        # self.net2 = nn.Linear(10, 5)
        self.net = torch.nn.Linear(4096, 150000)

    def forward(self, x):
        o= self.net(x);print('39 net',x.shape,o.shape)
        return o



# def demo_basic(local_world_size, local_rank):


#     # setup devices for this process. For local_world_size = 2, num_gpus = 8,
#     # rank 0 uses GPUs [0, 1, 2, 3] and
#     # rank 1 uses GPUs [4, 5, 6, 7].
#     n = torch.cuda.device_count() // local_world_size
#     device_ids = list(range(local_rank * n, (local_rank + 1) * n))

#     print(
#         f"[{os.getpid()}] rank = {dist.get_rank()}, "
#         + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
#     )

#     model = ToyModel().cuda(device_ids[0])
#     ddp_model = DDP(model, device_ids)

#     #loss_fn = nn.MSELoss()
#     loss_fn=ce if nnflag else fast_cross_entropy_loss
#     optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

#     labels = torch.randint(100, 200, size=[bsz*seql]).to(device_ids[0]) if nnflag else torch.randint(100, 200, size=[bsz,seql]).to(device_ids[0])
#     x = torch.randn(size=[bsz*seql, 4096], #requires_grad=True
#                     ) if nnflag else torch.randn(size=[bsz,seql, 4096])

#     #####
#     cnt=0
#     t1=time.time()
#     while cnt<10:
#         optimizer.zero_grad()
#         outputs = ddp_model(x)
#         #labels = torch.randn(20, 5).to(device_ids[0])

#         loss=loss_fn(outputs, labels);print('loss',loss)
#         loss.backward()
#         optimizer.step()
#         cnt+=1
#     print('82 time',time.time()-t1)

def demo_basic():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    model = ToyModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])

    loss_fn=ce if nnflag else fast_cross_entropy_loss
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    labels = torch.randint(100, 200, size=[bsz*seql]).to(device_id) if nnflag else torch.randint(100, 200, size=[bsz,seql]).to(device_id)
    x = torch.randn(size=[bsz*seql, 4096], #requires_grad=True
                    ) if nnflag else torch.randn(size=[bsz,seql, 4096])

    #####
    cnt=0
    t1=time.time()
    while cnt<10:
        optimizer.zero_grad()
        outputs = ddp_model(x)
        #labels = torch.randn(20, 5).to(device_ids[0])

        loss=loss_fn(outputs, labels);print('loss',loss)
        loss.backward()
        optimizer.step()
        cnt+=1
    print('82 time',time.time()-t1)

if __name__ == "__main__":
    demo_basic()