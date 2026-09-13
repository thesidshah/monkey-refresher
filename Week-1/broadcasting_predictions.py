"""
Broadcasting shape prediction drill.

Instructions:
1. For each expression below, WRITE DOWN your predicted output shape on paper
   (or in your head) BEFORE running this file or scrolling down.
2. Only after you've predicted all 5, run:  .venv/bin/python broadcasting_predictions.py
3. Answers are printed at the bottom — don't peek at the ANSWERS section in the
   source until you've committed to predictions.
"""
import torch

a = torch.zeros(3, 4)
b = torch.zeros(4)
# Q1: predict shape of (a + b)
# predicted: (3,4)

c = torch.zeros(5, 1, 4)
d = torch.zeros(1, 3, 4)
# Q2: predict shape of (c + d)
# predicted: (5,3,4)

e = torch.zeros(8, 1, 6, 1)
f = torch.zeros(7, 1, 5)
# Q3: predict shape of (e + f)
# Predicted: (8,1,6,1) - wrong. Rule: It should either be the same number of columns or 1. And if it's 1, the other number is selected.

g = torch.zeros(3, 1)
h = torch.zeros(3)
# Q4: predict shape of (g + h)
# Predicted: (3,1) - wrong.

i = torch.zeros(2, 3, 4)
j = torch.zeros(3, 1)
# Q5: predict shape of (i * j)
# Predicted: (2,3,4)

print("Write your 5 predictions before reading further.\n")
input("Press Enter once you've committed to all 5 predictions...")

print("\n--- ANSWERS ---")
print("Q1 (3,4)+(4,)      actual:", (a + b).shape)
print("Q2 (5,1,4)+(1,3,4) actual:", (c + d).shape)
print("Q3 (8,1,6,1)+(7,1,5) actual:", (e + f).shape)
print("Q4 (3,1)+(3,)      actual:", (g + h).shape)
print("Q5 (2,3,4)*(3,1)   actual:", (i * j).shape)
