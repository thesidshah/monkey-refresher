"""
20 short PyTorch tensor exercises. No solutions here on purpose.
Fill in each TODO, then run: .venv/bin/python tensor_exercises.py
Each block has an assert that self-checks your answer.
"""
import torch

# 1. Create a 1D tensor [0, 1, 2, ..., 9] using torch.arange.
t1 = torch.arange(0,10,1) 
assert t1.tolist() == list(range(10))

# 2. Create a 3x3 tensor of all ones with dtype=torch.float64.
t2 = torch.ones((3,3), dtype=torch.float64)  
assert t2.shape == (3, 3) and t2.dtype == torch.float64

# 3. Create a tensor with the same shape and dtype as t2, but filled with zeros,
#    using a *_like function.
t3 = torch.zeros_like(t2)
assert t3.shape == t2.shape and t3.dtype == t2.dtype and t3.sum() == 0

# 4. Create a numpy array [1,2,3], convert it to a tensor with torch.from_numpy,
#    then mutate the numpy array in place and show the tensor changed too.
import numpy as np
arr = np.array([1, 2, 3])
t4 = torch.from_numpy(arr)
arr[0] = 99
assert t4[0].item() == 99  # proves shared memory

# 5. Given x = torch.tensor([1, 2, 3]), cast it to float32 without using .to(...).
x5 = torch.tensor([1, 2, 3])
t5 = x5.float()
print(t5.dtype)
assert t5.dtype == torch.float32

# 6. Check what dtype results from adding an int64 tensor and a float32 tensor.
#    Store the resulting dtype in result_dtype (don't hardcode — compute it).
a6 = torch.tensor([1, 2], dtype=torch.int64)
b6 = torch.tensor([1.0, 2.0], dtype=torch.float32)
result_dtype = (a6+b6).dtype
assert result_dtype == torch.float32

# 7. Predict then verify: broadcast-add a (4,1) tensor and a (1,5) tensor.
#    Store the output shape as a tuple in shape7.
a7 = torch.zeros(4, 1)
b7 = torch.zeros(1, 5)
shape7 = (4,5)
assert tuple((a7 + b7).shape) == shape7

# 8. Create a tensor x = torch.arange(6), reshape it to (2,3) as a VIEW,
#    then confirm it shares storage with x via data_ptr().
x8 = torch.arange(6)
v8 = x8.view(2,3)  # TODO: reshape/view to (2,3)

# 9. Take a (4,4) tensor, slice out rows 1:3, and confirm the slice is a view
#    (mutating the slice mutates the original).
x9 = torch.arange(16).reshape(4, 4)
s9 = x9[1:3]  # TODO: slice rows 1:3
s9[0, 0] = -1
assert x9[1, 0].item() == -1

# 10. Take the same-shape tensor x9, index it with a boolean mask (x9 > 5),
#     and confirm the result is a COPY (mutating it does not change x9).
mask10 = x9 > 5

c10 = x9[mask10]  # TODO: boolean-indexed copy
original_val = x9[x9 > 5][0].item()
c10[0] = -999
assert x9[x9 > 5][0].item() == original_val

# 11. Use .clone() to make an explicit copy of a tensor and prove it's independent.
x11 = torch.tensor([1, 2, 3])
c11 = x11.clone()  # TODO
c11[0] = 100
assert x11[0].item() == 1

# 12. In-place add 5 to every element of a tensor (no reassignment, use the `_` op).
x12 = torch.tensor([1, 2, 3])
x12.add_(5) # TODO: in-place add
assert x12.tolist() == [6, 7, 8]

# 13. Given a non-contiguous tensor (from a transpose), call .reshape() on it
#     directly (not .view()) and confirm it still works, storing the result shape.
x13 = torch.arange(12).reshape(3, 4).t()  # non-contiguous, shape (4,3)
r13 = x13.reshape(2,6)  # TODO: reshape to (2, 6)
assert tuple(r13.shape) == (2, 6)

# 14. Permute a (2,3,4) tensor to shape (4,2,3) using .permute(), store in p14.
x14 = torch.rand(2, 3, 4)
p14 = x14.permute(2,0,1)  # TODO
assert tuple(p14.shape) == (4, 2, 3)

# 15. Try calling .view() (not .reshape()) directly on a transposed (non-contiguous)
#     tensor and catch the resulting RuntimeError. Set caught15 = True if it raises.
x15 = torch.arange(12).reshape(3, 4).t()
caught15 = False
# TODO: try x15.view(2, 6), except RuntimeError: set caught15 = True
try:
    x15.view(2,6)
except RuntimeError as e:
    caught15=True
assert caught15 is True

# 16. Use unsqueeze to turn a (5,) tensor into a (1,5) tensor, store in u16.
x16 = torch.arange(5)
u16 = torch.unsqueeze(x16,0)  # TODO
assert tuple(u16.shape) == (1, 5)

# 17. Use squeeze to remove all size-1 dims from a (1,3,1,4) tensor, store in sq17.
x17 = torch.rand(1, 3, 1, 4)
sq17 = torch.squeeze(x17)  # TODO
assert tuple(sq17.shape) == (3, 4)

# 18. Broadcast-multiply a (3,4) tensor by a scalar tensor torch.tensor(2.0),
#     store the result shape (should be unchanged) in shape18.
x18 = torch.rand(3, 4)
shape18 = tuple((x18 * torch.tensor(2.0)).shape)  # TODO
assert shape18 == (3, 4)

# 19. Given x = torch.arange(24).reshape(2,3,4), flatten it to 1D with .flatten(),
#     and confirm the number of elements is preserved.
x19 = torch.arange(24).reshape(2, 3, 4)
f19 = x19.flatten()  # TODO
assert f19.numel() == 24 and f19.dim() == 1

# 20. Demonstrate the in-place-vs-autograd trap: create x = torch.tensor([1.0],
#     requires_grad=True), then attempt x.add_(1) and catch the RuntimeError,
#     setting caught20 = True.
caught20 = False
x20 = torch.tensor([1.0], requires_grad=True)
# TODO: try x20.add_(1), except RuntimeError: set caught20 = True
try:
    x20.add_(1)
except RuntimeError:
    caught20=True
assert caught20 is True

print("All 20 exercises passed!")
