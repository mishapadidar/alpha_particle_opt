import numpy as np

def gauss_quadrature_nodes_coeffs(n_nodes=2,a=-1.0,b=1.0):
  """
  Return the roots and coefficients for gauss quadrature.

  With n_nodes nodes we can integrate any polynomial of degree
  2*n_nodes-1 exactly.

  Taken from Numerical Analysis, Burden and Faires

  n_nodes: number of quadrature nodes. 
  a,b: left and right hand side bounds of the integral.
  """
  assert n_nodes >= 2
  assert n_nodes <= 7

  roots = [[0.5773502692,-0.5773502692], # 2 nodes
           [0.7745966692,0.0000000000,-0.7745966692], # 3 nodes
           [0.8611363116,0.3399810436,-0.3399810436,-0.8611363116],
           [0.9061798459,0.5384693101,0.0000000000,-0.5384693101,-0.9061798459],
           [-0.9324695142031520278123,-0.661209386466264513661,-0.2386191860831969086305,0.238619186083196908631,0.661209386466264513661,0.9324695142031520278123],
           [-0.949107912342758525,-0.74153118559939444,-0.405845151377397167,0.0,0.405845151377397167,0.74153118559939444,0.949107912342758525]
           ]
  coeffs = [[1.0,1.0],
            [0.5555555556,0.8888888889,0.5555555556],
            [0.3478548451,0.6521451549,0.6521451549,0.3478548451],
            [0.2369268850,0.4786286705,0.5688888889,0.4786286705,0.2369268850],
            [0.1713244923791703450403,0.3607615730481386075698,0.4679139345726910473899,0.46791393457269104739,0.3607615730481386075698,0.1713244923791703450403],
            [0.129484966168869693,0.279705391489276668,0.381830050505118945,0.417959183673469388,0.38183005050511894,0.279705391489276668,0.129484966168869693]
           ]

  # transform to other interval
  nodes = ((b-a)*np.array(roots[n_nodes-2]) + (b+a))/2.0
  coeffs = np.array(coeffs[n_nodes-2])*(b-a)/2.0

  return nodes,coeffs


if __name__ == "__main__":

  f = lambda x : x**2 -2*x - 3
  fint = lambda x: x**3 / 3 - x**2 -3*x # integral
  a = -1.0
  b = 1.0
  true_int = fint(b) - fint(a)
  nodes,coeffs = gauss_quadrature_nodes_coeffs(n_nodes=2,a=a,b=b)
  fX = f(nodes)
  print(fX @ coeffs)
  print(true_int)
 
  f = lambda x : x**2 -2*x - 3
  fint = lambda x: x**3 / 3 - x**2 -3*x # integral
  a = -7.0
  b = 9.21
  true_int = fint(b) - fint(a)
  nodes,coeffs = gauss_quadrature_nodes_coeffs(n_nodes=2,a=a,b=b)
  fX = f(nodes)
  print(fX @ coeffs)
  print(true_int)


  f = lambda x : 13*x**7 -102.8*x**5 + x**2 -2*x - 14.
  fint = lambda x: 13*x**8 / 8 - 102.8*x**6/6 + x**3 / 3 - x**2 -14*x # integral
  a = -1.0
  b = 4.0
  true_int = fint(b) - fint(a)
  n_nodes = 4
  nodes,coeffs = gauss_quadrature_nodes_coeffs(n_nodes=n_nodes,a=a,b=b)
  fX = f(nodes)
  print(fX @ coeffs)
  print(true_int)


  f = lambda x : 13*x**12 -102.8*x**5 + x**2 -2*x - 14.
  fint = lambda x: 13*x**13 / 13 - 102.8*x**6/6 + x**3 / 3 - x**2 -14*x # integral
  a = -1.0
  b = 4.0
  true_int = fint(b) - fint(a)
  n_nodes = 7
  nodes,coeffs = gauss_quadrature_nodes_coeffs(n_nodes=n_nodes,a=a,b=b)
  fX = f(nodes)
  print(fX @ coeffs)
  print(true_int)

