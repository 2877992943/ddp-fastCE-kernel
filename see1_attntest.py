
import torch

### use
from sageattention import sageattn
import time


import torch.nn.functional as F


device='cuda'

 
#### sageAttn 输入
def sage_input(b=1,h=8,s=4096)
    hd=[64,128]
    #b,h,s,dim=2,32,4096,hd[-1]
    dim=hd[-1]
    q=torch.randn([b,h,s,dim] ,dtype=torch.float16).to(device)
    k=torch.randn([b,h,s,dim] ,dtype=torch.float16).to(device)
    v=torch.randn([b,h,s,dim] ,dtype=torch.float16).to(device)
    return q,k,v
#print(q.dtype) #float16
q11=torch.randn([b,s,h,dim] ,dtype=torch.float16).to(device)
k11=torch.randn([b,s,h,dim] ,dtype=torch.float16).to(device)
v11=torch.randn([b,s,h,dim] ,dtype=torch.float16).to(device)



def run1_sage2():
    from sageattention import sageattn
    q,k,v=sage_input()
    attn_output = sageattn(q, k, v, tensor_layout="HND", is_causal=False)




def memsee(tensor):
    # 确保你的张量是在CUDA设备上的
    tensor = tensor.to('cuda')

# 获取张量的显存占用大小
    tensor_size_in_bytes = tensor.element_size() * tensor.numel()

# 获取当前GPU上的显存占用（这个值可能包括其他张量的占用）
    memory_allocated = torch.cuda.memory_allocated(tensor.device)

# 获取当前GPU上的最大显存占用（这个值可能包括分配但未使用的显存）
    memory_reserved = torch.cuda.memory_reserved(tensor.device)

    print(f"张量的显存占用大小为: {tensor_size_in_bytes} bytes,{tensor_size_in_bytes//1024**2}m")
    print(f"当前GPU上的显存占用为: {memory_allocated} bytes,{memory_allocated//1024**2}m")
    print(f"当前GPU上的最大显存占用为: {memory_reserved} bytes,{memory_reserved//1024**2}m")


def run1():
    
    attn_output = sageattn(q, k, v, is_causal=True, smooth_k=True)
    print(attn_output.shape)
    return attn_output
     
    

def run2(): 
    """https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"""
    
    attn=F.scaled_dot_product_attention
    with torch.backends.cuda.sdp_kernel(enable_math=False):
        attn_output =attn(q,k,v, is_causal=True)
    print(attn_output.shape)
    return attn_output
    

def run3():# trition fla2
    causal=True
    sm_scale=0.5
    from attn1_trition import attention
    o=attention(q, k, v, causal, sm_scale).half()
    
    return o
def run4():
    """https://github.com/Dao-AILab/flash-attention"""
    
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    o=flash_attn_func(q11,k11,v11)
    
    return o


for _ in range(5):
    t1=time.time()
    o=run3()
    print(time.time()-t1)
    memsee(o)

## pluge
