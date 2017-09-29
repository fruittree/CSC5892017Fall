# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 08:32:34 2015
Brief demo of linear algebra in Numpy
@author: bxiao
"""
import numpy as np
from numpy.linalg import *

#==============================================================================
# Basic array creation 
#==============================================================================
# create an array
# create an array with a series of numbers
A = np.arange(12)
print A
# reshape
A.shape  = (3,4)
print A
# create an array from list
a = np.array([1, 4, 5, 8])
type(a)

# create 2D array
a = np.array([[1, 2, 3], [4, 5, 6]], float)
# Arrays can be reshaped using tuples that specify new dimensions. 
a = np.array(range(10), float)
a = a.reshape((5,2))
#
## other ways to create arrays
b=  np.zeros((2,3), dtype=float)
print b

## The eye function returns matrices with ones along the kth diagonal:
## creae an identity matrix
i = 4*np.identity(4, dtype=float)
print i

i_k = np.eye(4, k=1, dtype=float)
print i_k

A = np.array([[1,1],[0,1]])
B = np.array([[2,0],[3,4]])
print A
print B

## element wise product
C = A*B
print C
print '**'
#print C
C = np.dot(A,B)
print C


##==============================================================================
## Linear algebra
##==============================================================================
# 
a = np.array([[1.0, 2.0], [3.0, 4.0]])
print a
print '*'
print a.transpose()

print '*'
print inv(a)

#u = np.eye(2)  # unit 2 by 2 marix
#print u
#j = np.array([[0.0, -1.0], [1.0, 0.0]])
#print np.dot (j, j) # matrix product
#
##==============================================================================
## Linear equations
##==============================================================================
A = np.array([[1,2],[3,4]])
print A
b = np.array([10, 20])
print b
x = solve(A,b)
print '*'
print x
print A*x
 






