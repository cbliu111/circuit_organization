import torch
import torch.nn as nn
from analyzer import NNAnalyzer, flatten_params, unflatten_params
import numpy as np
import torch.nn.functional as F

neurons = 10
device = 'cpu'
input_dim = 100
output_dim = 10

mlp = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, neurons),
        nn.ReLU(),
        nn.Linear(neurons, neurons),
        nn.ReLU(),
        nn.Linear(neurons, output_dim)
    )
loss_fn = nn.CrossEntropyLoss(reduction='mean')
x = torch.randn(1, input_dim)
y = torch.randn(1, output_dim)


def get_loss(w, a, b):
    o = torch.func.functional_call(mlp, w, a)
    return loss_fn(o, b)


params = dict(mlp.named_parameters())
total_params = np.sum([p.numel() for k, p in params.items()])
h1 = torch.func.hessian(get_loss, argnums=0)(params, x, y)
r = torch.zeros((total_params, total_params), device=device, requires_grad=False)
d = []
for k, v in h1.items():
    s = params[k].numel()
    t = [vv.reshape(s, -1) for kk, vv in v.items()]
    t = torch.cat(t, dim=1)
    d.append(t)
r += torch.cat(d, dim=0)
h1 = r.detach().numpy()

def hvp(g, v, p):
    """
    Compute the Hessian-vector product for a given vector `v`.
    """
    hvp_result = torch.autograd.grad(g, p, grad_outputs=v, retain_graph=True)
    return torch.cat([h.view(-1) for h in hvp_result])


block_size = 2
total_params = np.sum([p.numel() for p in mlp.parameters()])
block_indices = []
blocks = total_params // block_size
if total_params % block_size > 0:
    blocks += 1
start_index = 0
for i in range(blocks):
    end_index = min(start_index + block_size, total_params)
    block_indices.append((start_index, end_index))
    start_index = end_index
hessian_blocks = []
for i, (start_i, end_i) in enumerate(block_indices):
    numel = min(block_size, end_i - start_i)
    hessian_blocks.append(torch.zeros((numel, total_params), device='cpu'))

params = [p for p in mlp.parameters() if p.requires_grad]
flat_params = flatten_params(params)
unflattened_params = unflatten_params(flat_params, params)
# identity_mat = torch.eye(block_size, device=self.device)
hessian_loss = 0
unflattened_params = unflatten_params(flat_params, params)
# 3 layer mlp in function form
y_hat = x.reshape(x.shape[0], -1) @ unflattened_params[0].T + unflattened_params[1]
y_hat = F.relu(y_hat)
y_hat = y_hat @ unflattened_params[2].T + unflattened_params[3]
y_hat = F.relu(y_hat)
y_hat = y_hat @ unflattened_params[4].T + unflattened_params[5]
hessian_loss = hessian_loss + loss_fn(y_hat, y).mean()
grads = torch.autograd.grad(hessian_loss, unflattened_params, create_graph=True)
grads = flatten_params(grads)
for i, (start_i, end_i) in enumerate(block_indices):
    # Batch of row vectors (one-hot vectors for specific rows, or random vectors)
    numel = min(block_size, end_i - start_i)
    row_vectors = torch.zeros((numel, total_params), device=device)
    row_vectors[:, start_i:end_i] = torch.eye(numel, device=device)  # One-hot vectors for selecting the rows
    hvp_block = torch.func.vmap(hvp, (None, 0, None))(grads, row_vectors, flat_params)
    hessian_blocks[i] += hvp_block.detach().cpu().numpy()
del grads
del hvp_block
h2 = np.concatenate(hessian_blocks, axis=0)

print(np.allclose(h1, h2))




