
import numpy as np

class EvalWrapper:
  """
  Class for wrapping a function call so that we can 
  save evaluations.
  F: function handle
  dim_x: function input dimension
  dim_F: function output dimension
  Usage:
    F = lambda x: [np.linalg.norm(x),np.sum(x)]
    dim_x = 3
    dim_F = 2
    func = eval_wrapper(f,dim_x,dim_F)
    # call the function
    func(np.random.randn(dim_x))
    # check the history
    print(func.X)
    print(func.FX)
  """

  def __init__(self,F,dim_x,dim_F):
    self.dim_x = dim_x
    self.dim_F = dim_F
    self.X = np.zeros((0,dim_x))
    self.FX = np.zeros((0,dim_F))
    self.func = F

  def __call__(self,xx):
    ff = self.func(xx)
    self.X = np.append(self.X,[xx],axis=0)
    if self.dim_F == 1:
      self.FX = np.append(self.FX,[[ff]],axis=0)
    else:
      self.FX = np.append(self.FX,[ff],axis=0)
    return ff
