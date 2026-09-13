# PyTorch Tensors — Core Concepts

## 1. Creation
- `torch.tensor(data)` — copies data, infers dtype
- `torch.zeros/ones/empty/full(shape)`
- `torch.arange(start, end, step)`, `torch.linspace(start, end, steps)`
- `torch.rand(shape)` (uniform [0,1)), `torch.randn(shape)` (standard normal)
- `torch.from_numpy(arr)` — **shares memory** with the numpy array (no copy)
- `*_like` variants: `torch.zeros_like(x)`, `torch.rand_like(x)` — same shape/dtype as `x`

## 2. dtype
- Default float dtype: `torch.float32`. Default int dtype: `torch.int64`.
- Set explicitly: `torch.tensor([1,2], dtype=torch.float64)`
- Cast: `x.to(torch.float32)`, `x.float()`, `x.long()`, `x.double()`
- Mixed-dtype ops upcast automatically (int + float -> float); mixing float32/float64 in matmul etc. will error in some ops — cast explicitly to be safe.
- Check with `x.dtype`.

## 3. Broadcasting
Two dims are compatible if they're equal, or one of them is 1 (or missing — shapes are aligned from the right).
- `(3,4) + (4,)` -> `(3,4)`
- `(3,1) + (1,4)` -> `(3,4)`
- `(5,3,4) + (3,4)` -> `(5,3,4)`
- Incompatible: `(3,4) + (3,)` (4 vs 3, neither is 1) -> error

## 4. Views vs Copies
- **View** (shares underlying storage, no copy): `.view()`, `.reshape()` (when possible), `.t()`, `.transpose()`, `.permute()`, basic slicing `x[1:3]`
- **Copy** (new storage): `.clone()`, fancy/boolean indexing, `.contiguous()` (only copies if not already contiguous), most out-of-place arithmetic ops
- Check shared storage: `x.data_ptr() == y.data_ptr()` or `x.storage().data_ptr()`
- `.view()` requires the tensor to be contiguous; `.reshape()` falls back to a copy if a view isn't possible.
- Modifying a view modifies the original tensor's data.

## 5. In-place ops
- Suffixed with `_`: `x.add_(1)`, `x.mul_(2)`, `x.relu_()`, `x[0] = 5`
- Mutate the tensor's underlying storage directly — no new allocation.
- **Caution**: in-place ops on a tensor that requires grad and is needed for backward will raise an autograd error ("a leaf Variable that requires grad is being used in an in-place operation" or version-counter mismatch on non-leaf tensors used in the graph).
- Also risky when other tensors hold a *view* of the same storage — they change too, silently.

## 6. reshape / permute
- `.reshape(*shape)` — same data, new logical shape, total elements must match; one dim can be `-1` (inferred)
- `.permute(*dims)` — reorders axes (like a generalized transpose); returns a *view*, but the result is typically non-contiguous, so `.reshape()` after `.permute()` often needs `.contiguous()` first (or use `.reshape()` which copies automatically if needed)
- `.transpose(dim0, dim1)` — swaps exactly two axes, also a view
- `.flatten()` / `.unsqueeze(dim)` / `.squeeze(dim)` — shape manipulation without touching data order (flatten may copy if non-contiguous)
