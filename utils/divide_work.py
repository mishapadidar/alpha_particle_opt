
def divide_work(n,k):
  """
  divide n numbers into k intervals
  as evenly as possible.

  args n, k
  n: int
  n: positive number
  k: int
  k: positive number

  return start, end
  intervals: 2D-array
  intervals: contains all numbers assigned to each 
             worker k.
  """

  # storage
  intervals = []
  counts = []

  # corner case
  if n <= k:
    intervals = [[ii] for ii in range(n)]
    counts = [1 for ii in range(n)]
    for ii in range(k-n):
      intervals.append([])
      counts.append(0)
  else:
    base     = n//k  
    leftover = n%k
    start    = 0   
    for ii in range(k): 
      if ii < leftover:
        size = base +1 
      else:
        size = base
      end   = start + size   
      # save the interval
      counts.append(size)
      intervals.append(range(start,end))
      # reset for next iteration
      start = end

  return intervals,counts

if __name__ == '__main__':

  print(divide_work(6,11))
  print(divide_work(2,5))
  print(divide_work(3,3))
  print(divide_work(10,3))
  print(divide_work(9,3))
  print(divide_work(4,2))
  print(divide_work(21,7))
