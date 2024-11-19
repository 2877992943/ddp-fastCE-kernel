
import torch

### use
from sageattention import sageattn
import time


import torch.nn.functional as F


device='cuda'

hd=[64,96,128]
b,h,s,dim=2,32,4096,hd[-1]
q=torch.randn([b,h,s,dim] ,dtype=torch.float16).to(device)
k=torch.randn([b,h,s,dim] ,dtype=torch.float16).to(device)
v=torch.randn([b,h,s,dim] ,dtype=torch.float16).to(device)
#print(q.dtype) #float16
q11=torch.randn([b,s,h,dim] ,dtype=torch.float16).to(device)
k11=torch.randn([b,s,h,dim] ,dtype=torch.float16).to(device)
v11=torch.randn([b,s,h,dim] ,dtype=torch.float16).to(device)

def run1():
    
    attn_output = sageattn(q, k, v, is_causal=True, smooth_k=True)
    print(attn_output.shape)

def run2(): 
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""
    
    attn=F.scaled_dot_product_attention
    with torch.backends.cuda.sdp_kernel(enable_math=False):
        attn_output =attn(q,k,v, is_causal=True)
    print(attn_output.shape)

def run3():# trition fla2
    causal=True
    sm_scale=0.5
    from attn1_trition import attention
    attention(q, k, v, causal, sm_scale).half()
    return None
def run4():
    """https://github.com/Dao-AILab/flash-attention"""
    
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    flash_attn_func(q11,k11,v11)
    return None


for _ in range(10):
    t1=time.time()
    run3()
    print(time.time()-t1)

## pluge