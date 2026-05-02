import numpy as np
from files import Neuron

rng = np.random.default_rng()

class RNNNode:
  def __init__(self , nin):
    self.wax = [Neuron(rng.random()) for _ in range(nin)]
    self.by = Neuron(rng.random())
    self.waa = [Neuron(rng.random()) for _ in range(nin)]
    self.ba = Neuron(rng.random())
    self.wya = Neuron(rng.random())

  def __call__(self,x,a_t_1):
    prodx = sum(xi*wi for xi , wi in zip(x,self.wax))
    proda = prodx + sum(ai*wi for ai , wi in zip(a_t_1,self.waa))
    a = proda + self.ba
    at = a.tanh()

    prody = at*self.wya + self.by
    yt = prody.tanh()

    return yt,at
  
class RNNlayer:
  def __init__(self , nin , nout):
    self.nodes = [RNNNode(nin) for n in range(nout)]

  def __call__(self , x , a):
    outy = []
    outa = []
    for n in self.nodes:
      y, a_new = n(x,a)

      outy.append(y)
      outa.append(a_new)

    return outy,outa

class RNN:
  def __init__(self , layers , nin):
    self.layers = layers
    self.a = [Neuron(rng.random()) for _ in range(nin)]

  def __call__(self , x):
    for layer in self.layers:
      x , self.a = layer(x,self.a)
    return x