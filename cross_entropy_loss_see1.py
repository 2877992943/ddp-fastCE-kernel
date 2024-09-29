# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import triton
import triton.language as tl
import torch
#from .utils import calculate_settings, MAX_FUSED_SIZE, triton_tanh
from utils import calculate_settings, MAX_FUSED_SIZE, triton_tanh

from transformers.models.llama.modeling_llama import logger




@triton.heuristics({"DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],})
@triton.jit
def _cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE     : tl.constexpr,
    BLOCK_SIZE     : tl.constexpr,
    DO_SOFTCAPPING : tl.constexpr,
    SOFTCAP        : tl.constexpr,
):
    """
        Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
        Pi = exp(xi) / sum(exp(xi))
        CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
             = -y [ x - log[sum(exp(x))] ]
             = y * (log[sum(exp(x))] - x)
        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        logsumexp is also stable
        Take    y =         log[sum(exp(x))]
           exp(y) =             sum(exp(x))
           exp(y) =             sum(exp(x - c)*exp(c)) Since e^(x-c)*e^c = e^x
           exp(y) =      exp(c)*sum(exp(x - c))
               y  = log(exp(c)*sum(exp(x - c)))
               y  = c + log[sum(exp(x - c))]
        This means we can set c = max(x) to make sure
        exp(x - c) always is exp(x - max(x)).
        This ensures exp(x - max(x))'s maximum is 1 as exp(0) = 1.
    """
    row_idx = tl.program_id(0)
    logits_ptr    += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr      += row_idx
    logsumexp_ptr += row_idx
    labels_ptr    += row_idx

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf"))
    # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if label_idx != -100:
        x = tl.load(logits_ptr + label_idx)
        # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
        if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
        loss = logsumexp - x.to(tl.float32)
    else:
        loss = 0.0
    tl.store(logsumexp_ptr, logsumexp)
    tl.store(loss_ptr, loss)
pass


@triton.heuristics({"DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],})
@triton.jit
def _chunked_cross_entropy_forward(
    logits_ptr, logits_row_stride,
    loss_ptr,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE     : tl.constexpr,
    N_CHUNKS       : tl.constexpr,
    BLOCK_SIZE     : tl.constexpr,
    DO_SOFTCAPPING : tl.constexpr,
    SOFTCAP        : tl.constexpr,
):
    """
        256K vocab divided in 4 chunks

        |-65536-| |-65536-| |-65536-| |-65536-|
        |-------| |-------| |-------| |-------|
        |-------| |-------| |-------| |-------|

        If y == 0: CE_i = 0
        If y == 1: CE_i = logsumexp - x

        Notice we can do logsumexp for each chunk and then
        logsumexp[chunk_sum(logsumexp)] == logsumexp

        chunk_sum = log[chunk_sum(logsumexp)]
                  = log[exp(logsumexp(a)) + ... + exp(logsumexp(z))]
                  = log[exp(log[sum(exp(a))]) + ... + exp(log[sum(exp(z))])]
                  = log[sum(exp(a)) + ... + sum(exp(z))]
                  = logsumexp(x)

        This means we can perform a logsumexp for each chunk, then do a
        final logsumexp reduction!

        Ie do: logsumexp(chunked_logsumexp) - x
    """
    row_idx   = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    logits_ptr    += row_idx * logits_row_stride.to(tl.int64)
    loss_ptr      += row_idx
    logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
    labels_ptr    += row_idx

    col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE

    label_idx = tl.load(labels_ptr).to(tl.int32)
    logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf"))
    # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
    if DO_SOFTCAPPING: logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

    logits = logits.to(tl.float32)
    c = tl.max(logits, 0)
    logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

    if chunk_idx == 0:
        # logsumexp(chunked_logsumexp) - x
        # Do the -x separately
        if label_idx != -100:
            x = tl.load(logits_ptr + label_idx).to(tl.float32)
            # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
            if DO_SOFTCAPPING: x = SOFTCAP * triton_tanh(x / SOFTCAP)
            loss = -1.0 * x.to(tl.float32)
        else:
            loss = 0.0
        tl.store(loss_ptr, loss)
    pass
    tl.store(logsumexp_ptr, logsumexp)
pass


@triton.heuristics({"DO_SOFTCAPPING": lambda args: args["DO_SOFTCAPPING"],})
@triton.jit
def _cross_entropy_backward(
    logits_ptr, logits_row_stride,
    dloss_ptr,   dloss_row_stride,
    logsumexp_ptr,
    labels_ptr,
    VOCAB_SIZE     : tl.constexpr,
    BLOCK_SIZE     : tl.constexpr,
    DO_SOFTCAPPING : tl.constexpr,
    SOFTCAP        : tl.constexpr,
):
    """
        CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
        dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

        From https://en.wikipedia.org/wiki/LogSumExp
        d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

        dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
        dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
        dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

        If y == 0: dC/dx = 0
        If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
        If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
    """
    row_idx   = tl.program_id(0)
    block_idx = tl.program_id(1)

    logits_ptr += row_idx * logits_row_stride.to(tl.int64)
    dloss_ptr  += row_idx *  dloss_row_stride
    col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < VOCAB_SIZE
    label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

    if label_idx != -100:
        dloss = tl.load(dloss_ptr)
    else:
        dloss = 0.0

    x = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf"))
    # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
    if DO_SOFTCAPPING:
        # d/dx [t * tanh(1/t * x)] = 1 - tanh^2(1/t * x)
        partial = triton_tanh(x / SOFTCAP)
        x = SOFTCAP * partial
    pass

    logsumexp = tl.load(logsumexp_ptr + row_idx)
    y = tl.exp(x.to(tl.float32) - logsumexp)
    y = tl.where(
        col_offsets == label_idx,
        y - 1.0, # exp(x - logsumexp) - 1
        y,       # exp(x - logsumexp)
    )

    if DO_SOFTCAPPING:
        # d/dx [t * tanh(1/t * x)] = 1 - tanh^2(1/t * x)
        y = y * (1.0 - partial*partial)
    pass

    # If y == 0: dC/dx = 0 ==> we already masked it to be = 0, so dloss = 0.
    tl.store(logits_ptr + col_offsets, dloss * y, mask = mask)
pass


MAX_FUSED_SIZE = 65536 # 2**16

class Fast_CrossEntropyLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, labels, logit_softcapping = 0):
        print(230,'forward')
        device1=logits.device
        n_rows, vocab_size = logits.shape

        div, mod = divmod(vocab_size, MAX_FUSED_SIZE) ;#print(236,div, mod,vocab_size,MAX_FUSED_SIZE)  # 2 18928 150000 65536
        n_chunks = div + (mod != 0)
        losses = torch.empty(n_rows, dtype = torch.float32, device = device1)

        DO_SOFTCAPPING = (logit_softcapping != 0)

        if n_chunks == 1:
            # For small vocabs <= 65336 like Llama, Mistral
            BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
            logsumexp = torch.empty(n_rows, dtype = torch.float32, device = device1)

            _cross_entropy_forward[(n_rows,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE     = vocab_size,
                BLOCK_SIZE     = BLOCK_SIZE,
                DO_SOFTCAPPING = DO_SOFTCAPPING,
                SOFTCAP        = logit_softcapping,
                num_warps      = num_warps,
            )
        else:
            # For large vocabs > 65336 like Gemma 256K
            logsumexp = torch.empty((n_rows, n_chunks,), dtype = torch.float32, device =device1)

            _chunked_cross_entropy_forward[(n_rows, n_chunks,)](
                logits, logits.stride(0),
                losses,
                logsumexp,
                labels,
                VOCAB_SIZE     = vocab_size,
                N_CHUNKS       = n_chunks,
                BLOCK_SIZE     = MAX_FUSED_SIZE,
                DO_SOFTCAPPING = DO_SOFTCAPPING,
                SOFTCAP        = logit_softcapping,
                num_warps      = 32,
            )
            # logsumexp(chunked_logsumexp) - x
            # Do the -x separately
            logsumexp = torch.logsumexp(logsumexp, dim = 1) # Row sum
            losses += logsumexp
            losses.masked_fill_(labels == -100, 0) # Don't forget to mask padding out!
        pass

        ctx.save_for_backward(logits, logsumexp, labels)
        ctx.DO_SOFTCAPPING    = DO_SOFTCAPPING
        ctx.logit_softcapping = logit_softcapping
        return losses
    pass

    @staticmethod
    def backward(ctx, dlosses):
        print(286,'backward')
        logits, logsumexp, labels = ctx.saved_tensors
        n_rows, vocab_size = logits.shape

        BLOCK_SIZE = 4096
        div, mod = divmod(vocab_size, BLOCK_SIZE)
        n_blocks = div + (mod != 0)

        _cross_entropy_backward[(n_rows, n_blocks,)](
            logits,   logits.stride(0),
            dlosses, dlosses.stride(0),
            logsumexp,
            labels,
            VOCAB_SIZE     = vocab_size,
            BLOCK_SIZE     = BLOCK_SIZE,
            DO_SOFTCAPPING = ctx.DO_SOFTCAPPING,
            SOFTCAP        = ctx.logit_softcapping,
            num_warps      = 8,
        )
        return logits, None, None,
    pass
pass


@torch._disable_dynamo
def fast_cross_entropy_loss(logits, labels, logit_softcapping = 0):
    """
    Arguments:
        logits: (batch, seq_len, vocab_size)
        labels: (batch, seq_len,)
    Returns:
        losses: float
    """
    batch, seq_len, d = logits.shape
    assert(labels.shape == (batch, seq_len))

    loss = Fast_CrossEntropyLoss.apply(
        logits.view(batch*seq_len, d),
        labels.view(-1),
        logit_softcapping,
    )
    n_items = torch.count_nonzero(labels != -100)
    return loss.sum() / n_items
pass




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



if __name__=='__main__':
    import time
    device = torch.device("cuda")if torch.cuda.is_available() else 'cpu'
    pred=torch.randn(size=[1024,8,150000],requires_grad=True).to(device)
    lab=torch.randint(100,200,size=[1024,8]).to(device)
    print (pred.device,lab.device)
    
    iter1=0
    
    while iter1<5:
        t1=time.time()
        loss=fast_cross_entropy_loss(pred,lab)
        t2=time.time()
        loss.backward()
        t3=time.time()
        print ( 'forward',t2-t1)
        print('backward', t3-t2) 
        iter1+=1
        memsee(pred)
    
    time.sleep(5)
    
    print('nn.ce....')
    
    import torch.nn.functional as nn
    ce=nn.cross_entropy 
    # Example of target with class indices
    
    iter2=0
    while iter2<5:
    
        pred = torch.randn(size=[1024*8, 150000], requires_grad=True).to(device)
        lab = torch.randint(100, 200, size=[1024*8, ]).to(device) 
        t1=time.time()
        loss = ce(pred,lab)
        t2=time.time()
        loss.backward()
        t3=time.time()
        print ( 'forward',t2-t1)
        print('backward', t3-t2) 
        iter2+=1
        memsee(pred)
        """
        tensor(14.7304, device='cuda:0', grad_fn=<DivBackward0>) 1.237128734588623
    tensor(14.7841, device='cuda:0', grad_fn=<NllLossBackward0>) 0.04305315017700195
        """