a = [[7, 387, 462, 450], [387, 25873, 28019, 25668],
     [462, 28019, 33388, 31090], [450, 25668, 31090, 31220]]

from alpha import *

a = Array(a)

inv_a = a.i

b = Array([469, 28194, 33104, 31842])

print(inv_a)
print("________________________________________________")
print(b)
print("________________________________________________")
# print(a.eig())
print("________________________________________________")
print(inv_a[0])
print("________________________________________________")
print("________________________________________________")
print(inv_a.dot(a))
print("________________________________________________")
print("________________________________________________")
print("________________________________________________")
print(inv_a@a)
print("________________________________________________")

print(a.det())
