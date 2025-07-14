from alpha import *


def elementwise_op(self, other=None, op=None):
    if isinstance(other, (int, float)):
        other = Array([[other]])
    elif isinstance(other, (list)):
        other = Array(other)
    elif not isinstance(other, Array):
        raise TypeError("Operand must be an Array or a number or a list")

    arr1, arr2 = Array._broadcast(self, other)

    def _elementwise_op(arr1, arr2, op):
        if len(arr1.shape) == 1:
            return Array([op(arr1.data[i], arr2.data[i]) for i in range(arr1.shape[0])])
        else:
            result_data = []
            for i in range(arr1.shape[0]):
                result_data.append(_elementwise_op(arr1[i], arr2[i], op).data)
            return Array(result_data)

    return _elementwise_op(arr1, arr2, op)




def min(self, axis=None):
    if axis is None:
        return min(self._flatten())
    else:
        data = self.data
        result = []
        for elem in data:
            result.append(self.min(elem, axis=axis-1))
        return min(result)

def max(self, axis=None):
    if axis is None:
        return max(self._flatten())
    else:
        data = self.data
        result = []
        for elem in data:
            result.append(self.max(elem, axis=axis-1))
        return max(result)



def clip(self, a_min, a_max):
    data = self.data
    result = []
    for elem in data:
        if elem < a_min:
            result.append(a_min)
        elif elem > a_max:
            result.append(a_max)
        else:
            result.append(elem)
    return Array(result)