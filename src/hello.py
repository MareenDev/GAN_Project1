print("Hello, nice to see you! Let's create awsome images!:)")
import torch
import random 
a = torch.ones(3)
print(a)
offset = 0.3
x = round(random.uniform(0, offset), 2)
y = round(random.uniform(0, offset), 2)	
print(x)
print(y)
b = a - x
c = a - y
print(b)
print(c)