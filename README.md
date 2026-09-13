# 🐒 monkey-refresher

*Give a monkey a GPU and it will `torch.zeros(8, 1, 6, 1)` something.*

This is a bag of small, self-checking drills for the moments when you open a
PyTorch file after three weeks away and your brain returns `NaN`. Every exercise
comes with an `assert`, so the repo tells you when you're wrong before a code
reviewer does.

No frameworks, no datasets, no 40-minute setup. Just tensors, a monkey, and the
questions that keep coming back.

## The kinds of problems in here

### 🍌 Shape prediction (the banana-counting drills)
You are shown two tensors and asked, *before running anything*, what shape
falls out. Broadcasting aligns from the right, not the left. The monkey learned
this by predicting `(8, 1, 6, 1)` and getting `(8, 7, 6, 5)`. The monkey is
still upset.

### 🪞 View or copy? (the mirror test)
Did `.reshape()` hand you the same memory or a fresh copy? Does a boolean mask
share storage? What does `.permute()` do to your strides, and why does
`.view()` suddenly throw a fit? The monkey pokes the tensor and watches whether
the original flinches.

### 🔧 The underscore ops (in-place grooming)
`add_`, `mul_`, `zero_`, `fill_`. The trailing underscore means "mutate me."
Great for saving memory. Terrible inside a forward pass with `requires_grad`.
There is an exercise where you trip over that on purpose.

### 🧮 Matrix multiplication by hand
Small enough to check on paper, big enough to remind you which dimension has
to match which. Comes with an HTML visualizer so you can watch the rows meet the
columns.

### 🐛 Trap exercises
A few drills exist purely to make you hit a `RuntimeError` and read it. Catching
the error is the answer. The monkey believes that reading error messages is a
skill, and that skills need reps.

## How to use it

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy
python Week-1/tensor_exercises.py
```

Each file either passes with a cheerful message or dies on the first wrong
answer. Fix it, rerun, repeat. Prediction files ask you to commit to an answer
in a comment before you scroll down. Honor system. The monkey is watching.

## Layout

```
Week-1/
  tensor_exercises.py           20 asserts about creation, dtype, views, in-place ops
  broadcasting_predictions.py   predict shapes first, then run
  broadcasting_viz_q3.html      why (8,1,6,1) + (7,1,5) is (8,7,6,5), interactive
  mat_mul_1.py                  matmul warmups in numpy
  mat_mul_1_viz*.html           row-meets-column animations
  tensor_notes.md               the cheat sheet the monkey keeps re-reading
private/                        git submodule, private repo, nothing to see here
```

## About that `private/` folder

It is a git submodule pointing at a private repo. You can see the pointer and
the URL, and that is all. Cloning it without access gives you an empty folder
and a polite refusal. It holds schedules and notes the monkey doesn't want
graded. It also exists so the monkey could learn how submodules work, which
turned out to be its own kind of refresher.

## Roadmap, loosely

- Week 1: tensors, shapes, memory, the things that bite
- Week 2: autograd, and why in-place ops and gradients don't mix
- Later: a tiny training loop from scratch, then losses, then whatever the monkey
  forgot most recently

## Contributing

If you have a drill that once made you feel dumb for twenty minutes and then
smart forever, it belongs here. Keep it small, keep the `assert`, and keep the
banana.

*No monkeys were harmed. Several tensors were.*
