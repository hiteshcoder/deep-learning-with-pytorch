# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:01:17 2020

@author: HiteshNayak
"""

import torch


#This is a 1-D Tensor or list
a = torch.tensor([2,2,1])
print(a)

#This is a 2-D Tensor or matrix
b = torch.tensor([[2,1,4],[3,5,4],[1,2,0],[4,3,2]])
print(b)

#The size of the tensors
print(a.shape)
print(b.shape)
print(a.size())
print(b.size())

#shape is an attribute and size is a method

#Get the height/number of rows of b
print(b.shape[0])

#float tensor
c = torch.FloatTensor([[2,1,4],[3,5,4],[1,2,0],[4,3,2]])
#or we can do
#c = torch.tensor([2,2,1], dtype = torch.float)

d = torch.DoubleTensor([[2,1,4],[3,5,4],[1,2,0],[4,3,2]])
#or we can do
#d = torch.tensor([2,2,1], dtype = torch.double)

print(c)
print(c.dtype)
print(d)
print(d.dtype)
#some statistical methods
print(c.mean())
print(d.mean())
print(c.std())
print(d.std())

#Reshape b 
#Note: If one of the dimensions is -1, its size can be inferred
print(b.view(-1,1))
print(b.view(12))
print(b.view(-1,4))
print(b.view(3,4))
#Assign b a new shape
b = b.view(1,-1)
print(b)
print(b.shape)
#We can even reshape 3D tensors
print('\n')
#Create a 3D Tensor with 2 channels, 3 rows and 4 columns (channles,rows,columns)
three_dim = torch.randn(2, 3, 4)
print('\n')
print(three_dim)
print(three_dim.view(2, 12))  # Reshape to 2 rows, 12 columns
print(three_dim.view(2, -1))

#Create a matrix with random numbers between 0 and 1
r = torch.rand(4,4)
print(r)

#Create a matrix with random numbers taken from a normal distribution with mean 0 and variance 1 
r2 = torch.randn(4,4)
print(r2)
print(r2.dtype)

#Create an array of 5 random integers from values between 6 and 9 (exlusive of 10)
in_array = torch.randint(6,10, (5,))
print(in_array)
print(in_array.dtype)

#Create a 2-D array (or matrix) of size 3x3 filled with random integers from values between 6 and 9 (exlusive of 10)
in_array2 = torch.randint(6,10, (3,3))
print(in_array2)

#Get the number of elemetns in in_array
print(torch.numel(in_array))
#Get the number of elemetns in in_array
print(torch.numel(in_array2))

#Construct a 3x3 matrix of zeros and of dtype long:
z = torch.zeros(3, 3, dtype=torch.long)
print(z)
#Construct a 3x3 matrix of ones
o = torch.ones(3,3)
print(o)
print(o.dtype)

r2_like = torch.randn_like(r2, dtype=torch.double)    # Convert the data type of the tensor
print(r2_like)

#Add two tensors, make sure they are the same size and data type
add_result = torch.add(r,r2)
print(add_result)

#In-place addition (change the value of r2)/ directly assign values 
r2.add_(r)    #r2=torch.add(r,r2)
print(r2)

#matrix slicing
#all rows and column 1
print(r2[:,1])
#all rows till 2nd column
print(r2[:,:2])

print(r2[:3,:])
num_ten = r2[2,3]
print(num_ten)
print(num_ten.item())
print(r2[2,:])


#NUMPY BRIDGE
import numpy as np

#Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
#See how the numpy array changed their value.
a.add_(1)
print(a)
print(b)
#if you want to change the values of a torch array it also affects the numpy arrays


#Converting NumPy Array to Torch Tensor
#See how changing the np array changed the Torch Tensor automatically
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

###FOR GPU PROCESSING########################
####if you have 
#Move the tensor to the GPU
r2 = r2.cuda()
print(r2)

#Provide Easy switching between CPU and GPU
CUDA = torch.cuda.is_available()
print(CUDA)
if CUDA:
    add_result = add_result.cuda()
    print(add_result)
    
#You can also convert a list to a tensor
a = [2,3,4,1]
print(a)
to_list = torch.tensor(a)
print(to_list, to_list.dtype)

data =  [[1., 2.], [3., 4.],
         [5., 6.], [7., 8.]]
T = torch.tensor(data)
print(T, T.dtype)

#tensor concatenation
#Tensor Concatenation 
first_1 = torch.randn(2, 5)
print(first_1)
second_1 = torch.randn(3, 5)
print(second_1)
#Concatenate along the 0 dimension (concatenate rows)
con_1 = torch.cat([first_1, second_1])
print('\n')
print(con_1)
print('\n')
first_2 = torch.randn(2, 3)
print(first_2)
second_2 = torch.randn(2, 5)
print(second_2)
# Concatenate along the 1 dimension (concatenate columns)
con_2 = torch.cat([first_2, second_2], 1)
print('\n')
print(con_2)
print('\n')

#adding dimensions to tensors
#Adds a dimension of 1 along a specified index
tensor_1 = torch.tensor([1, 2, 3, 4])
tensor_a = torch.unsqueeze(tensor_1, 0)
print(tensor_a)
print(tensor_a.shape)
tensor_b = torch.unsqueeze(tensor_1,1)
print(tensor_b)
print(tensor_b.shape)
print('\n')
#2 channels , 3 rows and 4 columns
tensor_2 = torch.rand(2,3,4)
print(tensor_2)
print('\n')
tensor_c = tensor_2[:,:,2]
print(tensor_c)
print(tensor_c.shape)
print('\n')
#this is assignment . Ask them to dig out whats this operation and where can it be used
tensor_d = torch.unsqueeze(tensor_c,2)
print(tensor_d)
print(tensor_d.shape)

#AUTOGRAD

#Remember, If requires_grad=True, the Tensor object keeps track of how it was created.
x = torch.tensor([1., 2., 3], requires_grad=True)
y = torch.tensor([4., 5., 6], requires_grad=True)
#Notice that both x and y have their required_grad set to true, therefore we can compute gradients with respect to them
z = x + y
print(z)
# z knows that is was created as a result of addition of x and y. It knows that it wasn't read in from a file
print(z.grad_fn)
#And if we go further on this
s = z.sum()
print(s)
print(s.grad_fn)

#Now if we backpropagate on s, we can find the gradients of s with respect to x
s.backward()
#gradient os s w.r.t x then we do this
print(x.grad)

# By default, Tensors have `requires_grad=False`
x = torch.randn(2, 2)
y = torch.randn(2, 2)
print(x.requires_grad, y.requires_grad)
z = x + y
# So you can't backprop through z
print(z.grad_fn)
#Another way to set the requires_grad = True is
x.requires_grad_()
y.requires_grad_()
# z contains enough information to compute gradients, as we saw above
z = x + y
print(z.grad_fn)
# If any input to an operation has ``requires_grad=True``, so will the output
print(z.requires_grad)
# Now z has the computation history that relates itself to x and y

#suppose I want to remove the relationship put upon z 
new_z = z.detach()
print(new_z.grad_fn)
# z.detach() returns a tensor that shares the same storage as ``z``, but with the computation history forgotten. 
#It doesn't know anything about how it was computed.In other words, we have broken the Tensor away from its past history

#You can also stop autograd from tracking history on Tensors. This concept is useful when applying Transfer Learning 
print(x.requires_grad)
print((x+10).requires_grad)

with torch.no_grad():
    print((x+10).requires_grad)
    
#Let's walk in through one last example
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)
z = y * y * 3
out = z.mean()
print(z, out)
out.backward()
print(x.grad)

m1 = torch.ones(5,5)
m2 = torch.zeros(5,5)
#Perform element-wise multiplaction 
mul = torch.mul(m1,m2)
#Another way to perform element-wise multiplaction 
mul_another = m1*m2
print(mul)
print(mul_another)

