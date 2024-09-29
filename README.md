# ddp-fastCE-kernel

this code try to integrate unsloth fast cross entropy kernal and multi gpu training with ddp


* code `cross_entropy_loss_see1.py` do things like seperate fast cross entropy kernal and estimate time and vram  

* code `ddp1-v1.py`   do multicard training
  

```

torchrun --nnodes=1 --nproc_per_node=2 ddp1-v1.py


```
