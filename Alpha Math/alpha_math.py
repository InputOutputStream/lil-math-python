import math
import copy
import random

class Array:

    def __init__(self, data):
        if not isinstance(data, list):
            raise TypeError("Data must be a list")
        self.data = data
        self.shape = self._calculate_shape()
        self.size = self._calculate_size()
        self.dtype = self._dtype()
        
    def copy(self):
        temp = copy.deepcopy(self.data)
        return Array(temp)

    def __repr__(self):
        return f'Array({self.data})'
    
    def __getitem__(self, key):
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return self.data[key]
        elif isinstance(key, list):
            return [self.data[i] for i in key]
        elif isinstance(key, tuple):
            if all(isinstance(k, int) for k in key):
                return self.data[key[0]][key[1]]
            elif all(isinstance(k, slice) for k in key):
                return [self.data[i] for i in range(*key)]
            elif all(isinstance(k, (int, slice)) for k in key):
                start, stop, step = key
                return self.data[start:stop:step]
            else:
                raise TypeError("Unsupported indexing type")
        elif isinstance(key, bool):
            return [x for i, x in enumerate(self.data) if key[i]]
        else:
            raise TypeError("Unsupported indexing type")


    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.data[key] = value
        elif isinstance(key, slice):
            self.data[key.start:key.stop:key.step] = value
        elif isinstance(key, list):
            for i, v in zip(key, value):
                self.data[i] = v
        elif isinstance(key, tuple):
            if all(isinstance(k, int) for k in key):
                self.data[key] = value
            elif all(isinstance(k, slice) for k in key):
                for i, v in zip(range(*key), value):
                    self.data[i] = v
            elif all(isinstance(k, (int, slice)) for k in key):
                start, stop, step = key
                for i, v in zip(range(start, stop, step), value):
                    self.data[i] = v
            else:
                raise TypeError("Unsupported indexing type")
        elif isinstance(key, bool):
            for i, v in enumerate(value):
                if key[i]:
                    self.data[i] = v
        else:
            raise TypeError("Unsupported indexing type")
        
    def __len__(self):
        return self.size
    
    def __containes__(self, x):
        try:
            self.data.index(x)   
            return True
        except:
            return False    
    def __ne__(self, other):
        return [x != other for x in self.data]
    
    def __eq__(self, other):        
        return [x == other for x in self.data]

    def __gt__(self, other):
        return [x > other for x in self.data]

    def __lt__(self, other):
        return [x < other for x in self.data]

    def __ge__(self, other):
        return [x >= other for x in self.data]

    def __le__(self, other):
        return [x <= other for x in self.data]

    def _dtype(self):
        if isinstance(self.data[0], list):  # Si c'est un tableau 2D
            return type(self.data[0][0])
        else:
            return type(self.data[0])
        
    def __str__(self):
        rows = self.shape[0]
        cols = self.shape[1] if rows > 0 else 0
        
        if rows == 0 or cols == 0:
            return "Array([])"

        if rows == 1:
            array_str = "("
            array_str += str(self.data)
            array_str += f")Array{self.shape}"
            return array_str

        array_str = "("
        for row in range(rows):
            array_str += "["
            for col in range(cols):
                array_str += str(self.data[row][col])
                if col < cols - 1:
                    array_str += ", "
            array_str += "]"
            if row < rows - 1:
                array_str += ",\n"
        array_str += f")Array{self.shape}"
        return array_str
    
    def __int__(self):
        if self.shape[0]>=2 and self.shape[1]>=2:
            result_data = [[int(self.data[i][j])  for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif self.shape[0] == 1:
            result_data = [ int(self.data[j]) for j in range(self.shape[1])] 
            return Array(result_data)
        elif self.shape[0] >= 2 and self.shape[1] == 1:
            result_data = [int(self.data[i])  for i in range(self.shape[0])]
            return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for : 'Array'")

    
    def __float__(self):
        if self.shape[0]>=2 and self.shape[1]>=2:
            result_data = [[float(self.data[i][j])  for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif self.shape[0] == 1:
            result_data = [ sqrt(self.data[j]) for j in range(self.shape[1])] 
            return Array(result_data)
        elif self.shape[0] >= 2 and self.shape[1] == 1:
            result_data = [sqrt(self.data[i])  for i in range(self.shape[0])]
            return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for : 'Array'")

    

    def _calculate_shape(self):
        def get_dimension_shape(arr):
            if isinstance(arr, list):
                return len(arr), *get_dimension_shape(arr[0])
            else:
                return ()
        
        shape = get_dimension_shape(self.data)
        if len(shape) == 1:  # If it's a 1D array
            return (1, shape[0])
        else:
            return shape
            
    def to_list(self):
        return self.data
    
    def to_ndarray(self):
        import numpy as np
        return np.array(self.data)
    
    @staticmethod
    def is_square_matrix(matrix):
        """
        Checks if a matrix is square.

        Args:
        - matrix: A 2D list representing the matrix.

        Returns:
        - A boolean value indicating whether the matrix is square or not.
        """
        return matrix.shape[0] == matrix.shape[1]

    @classmethod
    def from_ndarray(cls, ndarray):
        return cls(ndarray.tolist())
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    def __add__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[self.data[i][j] + other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [self.data[j] + other for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [[self.data[i][j] + other for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))
    def __radd__(self, other):
        return self + other

#.....................................................................................................
    def __sub__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[self.data[i][j] - other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [self.data[j] - other for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [[self.data[i][j] - other for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))
    def __rsub__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[other.data[i][j] - self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [ other - self.data[j] for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [ other - [self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))

#................................................................................................................


    def __mod__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[self.data[i][j] % other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [ self.data[j] % other for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [ [self.data[i][j] % other for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))

    def __rmod__(self, other):
        return self.__mod__(other)


#..........................................................................................................
    def __truediv__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[self.data[i][j] / other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [ self.data[j] / other for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [ [self.data[i][j] / other for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))

    def __rtruediv__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[other.data[i][j] / self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [ other / self.data[j] for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [[other / self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))
        
    def __itruediv__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    self.data[i][j]  /= other.data[i][j] 
            return self
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                for j in range(self.shape[1]):
                    self.data[i][j]  /= other.data[i][j] 
                return self
            elif self.shape[0] >= 2:
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        self.data[i][j]  /= other
                return self
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))

#.............................................................................................................    
    def __mul__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[other.data[i][j] * self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [ other * self.data[j] for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [[other * self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))

    # Reverse multiplication with scalar
    def __rmul__(self, other):
        return self.__mul__(other)  # Multiplication is commutative

#.............................................................................................................................
    def __floordiv__(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[self.data[i][j] // other.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [ self.data[j] // other  for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [[ self.data[i][j] // other for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))

    def __rfloordiv(self, other):
        if isinstance(other, Array):
            if self.shape != other.shape:
                raise ValueError("Shapes {} and {} are not aligned for addition".format(self.shape, other.shape))
            result_data = [[ other.data[i][j] // self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif isinstance(other, (int, float)):
            if self.shape[0] == 1:
                result_data = [ other // self.data[j]  for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2:
                result_data = [[ other // self.data[i][j] for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for +: 'Array' and '{}'".format(type(other)))


    def __matmul__(self, other):
        return self.matmul(other)

    def __pow__(self, n):
        if isinstance(n, (int, float)):
            if self.shape[0]>=2 and self.shape[1]>=2:
                result_data = [[math.pow(self.data[i][j], n)  for j in range(self.shape[1])] for i in range(self.shape[0])]
                return Array(result_data)
            if self.shape[0] == 1:
                result_data = [math.pow(self.data[j], n) for j in range(self.shape[1])] 
                return Array(result_data)
            elif self.shape[0] >= 2 and self.shape[1] == 1:
                result_data = [math.pow(self.data[i], n) for i in range(self.shape[0])]
                return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for *: 'Array' and '{}'".format(type(n)))

    def __neg__(self):
        return self * -1
    
    def __pos__(self):
        return self

    def __sqrt__(self):
        if self.shape[0]>=2 and self.shape[1]>=2:
            result_data = [[math.sqrt(self.data[i][j])  for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif self.shape[0] == 1:
            result_data = [ math.sqrt(self.data[j]) for j in range(self.shape[1])] 
            return Array(result_data)
        elif self.shape[0] >= 2 and self.shape[1] == 1:
            result_data = [ math.sqrt(self.data[i])  for i in range(self.shape[0])]
            return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for : 'Array'")

    
    
    def __round__(self):
        if self.shape[0]>=2 and self.shape[1]>=2:
            result_data = [[round(self.data[i][j])  for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif self.shape[0] == 1:
            result_data = [ round(self.data[j]) for j in range(self.shape[1])] 
            return Array(result_data)
        elif self.shape[0] >= 2 and self.shape[1] == 1:
            result_data = [ round(self.data[i])  for i in range(self.shape[0])]
            return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for : 'Array'")

    
    def __abs__(self):
        if self.shape[0]>=2 and self.shape[1]>=2:
            result_data = [[math.abs(self.data[i][j])  for j in range(self.shape[1])] for i in range(self.shape[0])]
            return Array(result_data)
        elif self.shape[0] == 1:
            result_data = [ math.abs(self.data[j]) for j in range(self.shape[1])] 
            return Array(result_data)
        elif self.shape[0] >= 2 and self.shape[1] == 1:
            result_data = [ math.abs(self.data[i])  for i in range(self.shape[0])]
            return Array(result_data)
        else:
            raise TypeError("Unsupported operand type(s) for : 'Array'")

        






#/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 
    
    @property
    def T(self):
        return self.transpose()

    @staticmethod
    def _broadcast(arr1, arr2):
        # Get the shapes of the arrays
        shape1 = arr1.shape
        shape2 = arr2.shape

        # Determine the maximum number of dimensions between the two arrays
        max_dim = max(len(shape1), len(shape2))

        # Extend the shapes of the arrays with leading ones to match dimensions
        new_shape1 = (1,) * (max_dim - len(shape1)) + shape1
        new_shape2 = (1,) * (max_dim - len(shape2)) + shape2

        # Broadcast the arrays along each dimension
        broadcasted_data1 = Array._broadcast_dimension(arr1.data, new_shape1)
        broadcasted_data2 = Array._broadcast_dimension(arr2.data, new_shape2)

        return broadcasted_data1, broadcasted_data2, max_dim

    @staticmethod
    def _broadcast_dimension(data, shape):
        # Initialize a list to hold the broadcasted data
        broadcasted_data = []

        # Broadcast the data along the dimension
        for dim in shape:
            if dim == 1:
                broadcasted_data.append([data[0]] * len(broadcasted_data[-1]))
            else:
                broadcasted_data.append(data)

        return broadcasted_data

    def _calculate_size(self):
        x = 1
        for i in range(len(self.shape)):
            x = x* self.shape[i]
        return x
    
    # @staticmethod
    # def _broadcast(arr1, arr2):
    #     if len(arr1) == len(arr2):
    #         return zip(arr1, arr2)
    #     elif len(arr1) < len(arr2):
    #         return zip(arr1 * len(arr2), arr2)
    #     else:
    #         return zip(arr1, arr2 * len(arr1))
#..........................................................................................................................................
    
    
    

    
    # Operations Mathematiques
    def sum(self, axis=None):
            if axis is None:
                return sum(sum(row) for row in self.data)
            elif axis == 0:
                return [sum(col) for col in zip(*self.data)]
            elif axis == 1:
                return [sum(row) for row in self.data]
    
    
  
    # Operations sur les arrays
    def matmul(self, matrix2):
        if self.shape[1] != matrix2.shape[0]:
            raise ValueError("Shapes {} and {} are not aligned".format(self.shape, matrix2.shape))

        result_data = []
        for row in self.data:
            new_row = []
            for col in zip(*matrix2.data):
                element = sum(a * b for a, b in zip(row, col))
                new_row.append(element)
            result_data.append(new_row)

        return Array(result_data)

    def dot(self, other):
        if isinstance(other, Array):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Shapes {} and {} are not aligned".format(self.shape, other.shape))
            result = [[sum(a * b for a, b in zip(row, col)) for col in zip(*other.data)] for row in self.data]
            return Array(result)
        else:
            raise TypeError("Unsupported operand type(s) for matrix multiplication: 'Array' and '{}'".format(type(other)))


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
        
    @staticmethod
    def zeros(shape):
        return Array([[0] * shape[1] for _ in range(shape[0])])
    
    def transpose(self):
        if self.shape[0] == 1:  # If 1D array, invert the shape
            return self._transpose_()
        elif self.shape[0] >= 2 and self.shape[1] >= 2:  # If 2D array, perform transpose
            transposed_data = list(map(list, zip(*self.data)))
            return Array(transposed_data)
        elif self.shape[0] >= 2 and self.shape[1] == 1:
            transposed_data = list(map(list, zip(*self.data)))
            return Array(transposed_data)
        else:
            print("................", self.shape)
            raise ValueError("Transpose is not defined for arrays with more than 2 dimensions")
        
    def _transpose_(self):
        if self.shape[1]>= 2 and self.shape[0] == 1:  # If it's a 1D array with shape (1, n)
            new_data = [[elem] for elem in self.data]  # Reshape the data
            return Array(new_data)
       
    @staticmethod
    def ones(shape):
        return Array([[1] * shape[1] for _ in range(shape[0])])
    
    @staticmethod
    def eye(shape):
        return Array([[int(j == i) for j in range(shape[1])] for i in range(shape[0])])
    
    @staticmethod
    def linspace(start, end, endpoint=True, shape=None):
        # Determine the number of elements in the array
        if shape is None:
            raise ValueError("Shape must be specified for linspace method")
        num_elements = shape[0] * shape[1]

        # Calculate the step size based on the number of elements
        if endpoint:
            step = (end - start) / (num_elements - 1)
        else:
            step = (end - start) / num_elements

        # Generate array data
        data = []
        val = start
        for _ in range(shape[0]):
            row = []
            for _ in range(shape[1]):
                row.append(val)
                val += step
            data.append(row)

        return Array(data)

    @staticmethod
    def arange(start, stop, step, shape=None):
        # Determine the number of elements in the array
        if shape is None:
            raise ValueError("Shape must be specified for arange method")
        num_elements = shape[0] * shape[1]

        # Generate array data
        data = []
        val = start
        for _ in range(shape[0]):
            row = []
            for _ in range(shape[1]):
                row.append(val)
                val += step
            data.append(row)

        return Array(data)  


#..........................................................................................................................................
    def round_values(self, decimals=2):
        rounded_data = [[round(val, decimals) for val in row] for row in self.data]
        return Array(rounded_data)
    
    def array_equal(self, other):
        if not isinstance(other, Array):
            raise TypeError("Unsupported operand type(s) for array_equal: 'Array' and '{}'".format(type(other)))
        if self.shape != other.shape:
            return False
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.data[i][j] != other.data[i][j]:
                    return False
        return True

    def argmax(self, axis=None):
        if axis is None:
            flattened_data = [val for row in self.data for val in row]
            max_val = max(flattened_data)
            return flattened_data.index(max_val)
        elif axis == 0:
            max_indices = Array([max(enumerate(col), key=lambda x: x[1])[0] for col in zip(*self.data)])
            return max_indices
        elif axis == 1:
            max_indices = Array([row.index(max(row)) for row in self.data])
            return max_indices
        else:
            raise ValueError("Invalid axis value. Axis must be None, 0, or 1.")

    def argmin(self, axis=None):
        if axis is None:
            flattened_data = [val for row in self.data for val in row]
            min_val = min(flattened_data)
            return flattened_data.index(min_val)
        elif axis == 0:
            min_indices = Array([min(enumerate(col), key=lambda x: x[1])[0] for col in zip(*self.data)])
            return min_indices
        elif axis == 1:
            min_indices = Array([row.index(min(row)) for row in self.data])
            return min_indices
        else:
            raise ValueError("Invalid axis value. Axis must be None, 0, or 1.")


    def reshape(self, shape):
        if self.size != shape[0] * shape[1]:
            raise ValueError("Cannot reshape array of size {} into shape {}".format(len(self.data), shape))
        return Array([self.data[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])])

    def repeat(self, repeats, axis=None):
        if axis is None:
            return Array([val for val in self.data for _ in range(repeats)])
        elif axis == 0:
            return Array([val for val in self.data for _ in range(repeats)])
        elif axis == 1:
            return Array([[val] * repeats for val in self.data])

    def concat(self, other, axis=0):
        a = self.copy()
        if axis == 0:
            return Array(a.data + other.data)
        elif axis == 1:
            if a.shape[0] != other.shape[0]:
                raise ValueError("Shapes {} and {} are not aligned".format(a.shape, other.shape))
            return Array([row + other.data[i] for i, row in enumerate(a.data)])

    def vstack(self, other):
        return self.concat(other, axis=0)

    def hstack(self, other):
        return self.concat(other, axis=1)
    
    
    def min(self, axis=None):
        if axis is None:
            return min(min(row) for row in self.data)
        elif axis == 0:
            return Array([min(col) for col in zip(*self.data)])
        elif axis == 1:
            return Array([min(row) for row in self.data])

    def max(self, axis=None):
        if axis is None:
            return max(max(row) for row in self.data)
        elif axis == 0:
            return Array([max(col) for col in zip(*self.data)])
        elif axis == 1:
            return Array([max(row) for row in self.data])
        
    def isclose(self, other):
        print(self.shape, Array.is_square_matrix(self))
        if not Array.is_square_matrix(self):
            raise ValueError("Cannot perform Gram-Schmidt process: array is not square")
        return Array([math.isclose(x, y) for x, y in zip(self.data, other)])

    def flatten(self, axis=None):
        if axis is None:
            # Default axis to flatten entire array
            return self.__flatten_all__()
        elif axis == 0:
            # Flatten along rows (axis 0)
            return self.__flatten_rows__()
        elif axis == 1:
            # Flatten along columns (axis 1)
            return self.__flatten_columns__()
        else:
            print("axis = ", axis)
            raise ValueError("Invalid axis. Please specify axis as 0 or 1.")

    def __flatten_all__(self):
        flattened_data = [val for row in self.data for val in row]
        return Array(flattened_data)

    def __flatten_rows__(self):
        flattened_data = [val for row in self.data for val in row]
        return Array([flattened_data])

    def __flatten_columns__(self):
        flattened_data = [[val] for row in self.data for val in row]
        return Array(flattened_data)
    
    def ravel(self):
        return self.flatten()

    def sort(self):
        temp = self.data.copy()
        return Array([sorted(row) for row in temp])

    def clip(self, a_min, a_max):
        return Array([[max(min(val, a_max), a_min) for val in row] for row in self.data])

    def copy(self):
        return Array([row[:] for row in self.data])

    def reshape(self, new_shape):
        if math.prod(new_shape) != math.prod(self.shape):
            raise ValueError("Cannot reshape array of size {} into shape {}".format(math.prod(self.shape), new_shape))
        flat_data = self.flatten()
        return Array([[flat_data.pop(0) for _ in range(new_shape[1])] for _ in range(new_shape[0])])
    def resize(self, new_shape):
        if self.size() != new_shape[0] * new_shape[1]:
            raise ValueError("Cannot resize array to new shape: sizes do not match")
        flat_data = [val for row in self.data for val in row]
        new_data = [flat_data[i:i+new_shape[1]] for i in range(0, len(flat_data), new_shape[1])]
        return Array(new_data)

    def squeeze(self):
        if self.shape[0] == 1 and self.shape[1] > 1:
            return Array([val for val in self.data[0]])
        elif self.shape[0] > 1 and self.shape[1] == 1:
            return Array([val for row in self.data for val in row])
        else:
            return self

    def partition(self, kth, axis=-1):
        if axis == -1:
            sorted_data = sorted(self.data)
            return Array(sorted_data[:kth]), Array(sorted_data[kth:])
        else:
            raise NotImplementedError("Partition along axis other than -1 is not implemented yet")

    def argpartition(self, kth, axis=-1):
        if axis == -1:
            sorted_indices = sorted(range(len(self.data)), key=lambda i: self.data[i])
            return sorted_indices[:kth], sorted_indices[kth:]
        else:
            raise NotImplementedError("Argpartition along axis other than -1 is not implemented yet")

    def swapaxes(self, axis1, axis2):
        if axis1 not in (0, 1) or axis2 not in (0, 1):
            raise ValueError("Axis values must be 0 or 1")
        if axis1 == axis2:
            return self
        else:
            return Array(list(zip(*self.data)))

    def put(self, indices, values):
        flat_data = [val for row in self.data for val in row]
        for index, value in zip(indices, values):
            flat_data[index] = value
        return Array([flat_data[i:i+self.shape[1]] for i in range(0, len(flat_data), self.shape[1])])

    def choose(self, choices):
        return Array([choices[val] for row in self.data for val in row])

    def take(self, indices, axis=None):
        if axis is None:
            return Array([self.data[index] for index in indices])
        else:
            raise NotImplementedError("Take along axis other than None is not implemented yet")


#------------------------------------------------------------------------------------------------------------------------

   
    @staticmethod
    def array_equal(arr, other):
        if not isinstance(other, Array):
            raise TypeError("Unsupported operand type(s) for array_equal: 'Array' and '{}'".format(type(other)))
        if arr.shape != other.shape:
            return False
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr.data[i][j] != other.data[i][j]:
                    return False
        return True

 
#$$$$$$$$$ Random Muodule $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

    @staticmethod
    def _random_value():
        return random.random()

    @staticmethod
    def random(shape):
        if not isinstance(shape, tuple) or not all(isinstance(dim, int) for dim in shape):
            raise ValueError("Shape must be a tuple of integers")
        return Array([[Array._random_value() for _ in range(shape[1])] for _ in range(shape[0])])

    def randint(self, low, high, shape=None):
        if shape is None:
            return Array([[random.randint(low, high) for _ in range(self.shape[1])] for _ in range(self.shape[0])])
        elif isinstance(shape, tuple) and len(shape) == 2:
            return Array([[random.randint(low, high) for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            raise ValueError("Shape must be a tuple of two integers")
        
    @staticmethod
    def rand(low, high, shape=None):
        if shape is None:
            raise ValueError("Shape must be specified.")
        return Array([[random.random() * (high - low) + low for _ in range(shape[1])] for _ in range(shape[0])])

    def randn(self, shape=None):
        if shape is None:
            return Array([[random.gauss(0, 1) for _ in range(self.shape[1])] for _ in range(self.shape[0])])
        elif isinstance(shape, tuple) and len(shape) == 2:
            return Array([[random.gauss(0, 1) for _ in range(shape[1])] for _ in range(shape[0])])
        else:
            raise ValueError("Shape must be a tuple of two integers")

    def shuffle(self, axis=0):
        if axis != 0:
            raise ValueError("Cannot shuffle array along axis other than 0.")
        
        if self.shape[0] != 1:
            raise ValueError("Cannot shuffle array: array must be one-dimensional along the specified axis.")
        
        shuffled_data = self.data[0][:]
        random.shuffle(shuffled_data)
        return Array([shuffled_data])


    def choice(self):
        if self.shape[0] != 1:
            raise ValueError("Cannot choose element from array: array must be one-dimensional")
        return random.choice(self.data)
    
    def ndim(self):
        return len(self.shape)

    def size(self):
        return self.size

    def around(self, decimals=0):
        return Array([[round(val, decimals) for val in row] for row in self.data])

    def round(self, decimals=0):
        return self.around(decimals)

    def ptp(self):
        return self.amax() - self.amin()

    def all(self):
        return all(val for row in self.data for val in row)

    def any(self):
        return any(val for row in self.data for val in row)

    def compress(self, condition):
        return Array([[val for val in row if condition(val)] for row in self.data])
    
    def amax(self):
        max_val = float('-inf')
        for row in self.data:
            max_val = max(max(row), max_val)
        return max_val

    def prod(self):
        product = 1
        for row in self.data:
            product *= math.prod(row)
        return product

    def nonzero(self):
        indices = []
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                if self.data[i][j] != 0:
                    indices.append((i, j))
        return indices

    def clip(self, a_min=None, a_max=None):
        clipped_data = [[max(min(val, a_max), a_min) if a_min is not None else min(val, a_max) if a_max is not None else val for val in row] for row in self.data]
        return Array(clipped_data)

    def squeeze(self):
        squeezed_data = [val for row in self.data for val in row]
        return Array([squeezed_data])
    
    def amin(self):
        min_val = float('inf')
        for row in self.data:
            min_val = min(min(row), min_val)
        return min_val
    def diagonal(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Array must be square for diagonal")
        return [self.data[i][i] for i in range(self.shape[0])]

    def trace(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Array must be square for trace")
        return sum(self.diagonal())

    def searchsorted(self, values):
        return [[sorted(values).index(val) for val in row] for row in self.data]

    
    
    
    
    
    
    
#........je flex encore.................................................................................................................................

    # Fourier Transform
    def _fft(self, x):
        N = len(x)
        if N <= 1:
            return x
        even = self._fft(x[::2])
        odd = self._fft(x[1::2])
        min_length = min(len(odd), N // 2)  # Ensure range does not exceed length of odd
        T = [math.e ** (-2j * math.pi * k / N) * odd[k] for k in range(min_length)]
        return [even[k] + T[k] for k in range(min_length)] + [even[k] - T[k] for k in range(min_length)]
    def fft(self):
        if self.shape[0] != 1:
            raise ValueError("Cannot perform FFT: array must be one-dimensional")
        return Array(self._fft(self.data))
    
    

#....................................................................................................................................    
#'''''''''''Statistics''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    def mean(self, axis=None):
        if axis is None:
            return self.sum() / (self.shape[0] * self.shape[1])
        else:
            return [sum(row) / len(row) for row in self.data] if axis == 1 else [sum(col) / len(col) for col in zip(*self.data)]


    def std(self, axis=None):
        mu = self.mean(axis)
        if axis is None:
            sq_diff = sum((val - mu) ** 2 for row in self.data for val in row)
            return math.sqrt(sq_diff / (self.shape[0] * self.shape[1]))
        else:
            return [math.sqrt(sum((val - mu[i]) ** 2 for val in row) / len(row)) for i, row in enumerate(self.data)] if axis == 1 else [math.sqrt(sum((val - mu[i]) ** 2 for val in col) / len(col)) for i, col in enumerate(zip(*self.data))]

    def var(self, axis=None):
        return [std ** 2 for std in self.std(axis)]
        
    def cumsum(self, axis=None):
        if axis is None:
            cum_sum = []
            total = 0
            for row in self.data:
                cum_row = []
                for val in row:
                    total += val
                    cum_row.append(total)
                cum_sum.append(cum_row)
                total = 0
            return cum_sum
        elif axis == 0:
            cum_sum = []
            total = [0] * len(self.data[0])
            for row in self.data:
                cum_row = []
                for i, val in enumerate(row):
                    total[i] += val
                    cum_row.append(total[i])
                cum_sum.append(cum_row)
            return cum_sum
        elif axis == 1:
            cum_sum = []
            total = 0
            for row in self.data:
                cum_row = []
                for val in row:
                    total += val
                    cum_row.append(total)
                cum_sum.append(cum_row)
                total = 0
            return cum_sum

    def cumprod(self, axis=None):
        if axis is None:
            cum_prod = []
            prod = 1
            for row in self.data:
                cum_row = []
                for val in row:
                    prod *= val
                    cum_row.append(prod)
                cum_prod.append(cum_row)
            return cum_prod
        elif axis == 0:
            cum_prod = []
            prod = [1] * len(self.data[0])
            for row in self.data:
                cum_row = []
                for i, val in enumerate(row):
                    prod[i] *= val
                    cum_row.append(prod[i])
                cum_prod.append(cum_row)
            return cum_prod
        elif axis == 1:
            cum_prod = []
            for row in self.data:
                prod = 1
                cum_row = []
                for val in row:
                    prod *= val
                    cum_row.append(prod)
                cum_prod.append(cum_row)
            return cum_prod
        
#-------------Linear Algebra----------------------------------------------------------------------------------------------------------------------
    
    def norm(self, axis=None, distance='euclidean'):
        # if axis is None:
        #     axis = self.shape[0]

        if distance == 'euclidean':
            return self.euclidean_norm(axis)
        elif distance == 'manhattan':
            return self.manhattan_norm(axis)
        else:
            raise ValueError("Invalid distance type. Please choose 'euclidean' or 'manhattan'.")

    def euclidean_norm(self, axis):
        squared_sum = sum(val ** 2 for val in self.flatten(axis))
        return math.sqrt(squared_sum)

    def manhattan_norm(self, axis):
        return sum(abs(val) for val in self.flatten(axis))

    @property
    def inv(self):
        """
        Calculate the inverse of a square self using Gaussian elimination.
        
        Args:
        - self (list of lists): The input square self represented as a list of lists.
        
        Returns:
        - list of lists: The inverse of the input self.
        """
        if not self.is_square_matrix(self):
            raise ValueError("Input self must be square")

        n = self.shape[0]
        augmented_self = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(self)]

        # Forward elimination
        for i in range(n):
            # Find pivot row
            pivot_row = max(range(i, n), key=lambda j: abs(augmented_self[j][i]))
            if augmented_self[pivot_row][i] == 0:
                raise ValueError("self is singular")
            # Swap rows
            augmented_self[i], augmented_self[pivot_row] = augmented_self[pivot_row], augmented_self[i]
            # Scale pivot row
            pivot_val = augmented_self[i][i]
            augmented_self[i] = [val / pivot_val for val in augmented_self[i]]
            # Eliminate other rows
            for j in range(i + 1, n):
                multiplier = augmented_self[j][i]
                augmented_self[j] = [augmented_self[j][k] - multiplier * val for k, val in enumerate(augmented_self[i])]

        # Backward elimination
        for i in range(n - 1, -1, -1):
            for j in range(i - 1, -1, -1):
                multiplier = augmented_self[j][i]
                augmented_self[j] = [augmented_self[j][k] - multiplier * val for k, val in enumerate(augmented_self[i])]

        return Array([row[n:] for row in augmented_self])

    def allclose(self, arr2, rtol=1e-05, atol=1e-08):
        """
        Compare two arrays element-wise within a tolerance.

        Parameters:
        - self (array-like): First array to compare.
        - arr2 (array-like): Second array to compare.
        - rtol (float): The relative tolerance parameter (default: 1e-05).
        - atol (float): The absolute tolerance parameter (default: 1e-08).

        Returns:
        - bool: True if all elements are within the specified tolerance, False otherwise.
        """
        if len(self) != len(arr2):
            return False

        for x, y in zip(self, arr2):
            if abs(x - y) > atol + rtol * max(abs(x), abs(y)):
                return False

        return True
    
    def det(self):
        if not Array.is_square_matrix(self):
            raise ValueError("Cannot compute determinant: array is not square")
        
        determinant = 1.0
        arr_copy = self.copy()
        for j in range(self.shape[0]):
            max_row = j
            max_val = arr_copy.data[j][j]

            # Find the maximum absolute value element in column j from rows j to n
            for i in range(j + 1, self.shape[0]):
                if arr_copy.data[i][j] > max_val:
                    max_val = arr_copy.data[i][j]
                    max_row = i

            # Swap the row with maxVal with the current row
            if max_row != j:
                arr_copy.data[[j, max_row], :] = arr_copy.data[[max_row, j], :]
                # Multiply the determinant by -1 due to row swap
                determinant *= -1

            # Divide the current row by the value of A[j][j] to make the leading coefficient 1
            pivot = arr_copy.data[j][j]
            arr_copy.data[j, :] /= pivot
            determinant *= pivot

            # Eliminate the jth column element in other rows
            for i in range(j + 1, self.shape[0]):
                factor = arr_copy.data[i][j]
                arr_copy.data[i, :] -= factor * arr_copy.data[j, :]

        # Multiply the determinant by the diagonal elements of the row-echelon form
        for i in range(self.shape[0]):
            determinant *= arr_copy.data[i][i]

        return determinant
            
        
    def qr_decomposition(self):
        n = self.shape[0]  # Assuming square matrix
        q = Array.eye((n,n))  # Initialize Q as the identity matrix
        r = Array(self.data)  # Initialize R as a copy of the input matrix

        max_iterations = 1000
        tolerance = 1e-9

        for _ in range(max_iterations):
            # Iterate over columns of R
            for j in range(n):
                # Extract the j-th column of R
                v = [r.data[i][j] for i in range(j, n)]
                # Compute the norm of v and adjust the sign if necessary
                norm_v = sum([math.pow(x,2) for x in v])
                norm_v = math.sqrt(norm_v)
                
                if v[0] < 0:
                    norm_v = -norm_v
                
                # Check for convergence
                if norm_v < tolerance:
                    return q, r
                
                # Create the Householder reflector
                v[0] += norm_v
                scale = sum([math.pow(x,2) for x in v])
                scale = math.sqrt(scale)
                
                reflector = [x / scale for x in v]
                reflector = Array(reflector)

                # Update R and Q
                for i in range(j, n):
                    print(len(r.data[j]))
                    print(reflector.shape)
                    dot_product = (r.dot(reflector.T)).sum()          #@for k in range(j, n))
                    r.data[i][j] = dot_product
                    for k in range(n):
                        q.data[i][k] -= 2 * reflector[k] * dot_product


        return q, r

    def gram_schmidt(self):
        if self.shape[0] != self.shape[1]:
            raise ValueError("Cannot perform Gram-Schmidt process: array is not square")
        
        n = self.shape[0]
        u = Array(self.data)  # Copie des vecteurs d'origine
        e = Array([[0] * n for _ in range(n)])  # Matrice des vecteurs orthogonaux
        
        for j in range(n):
            # Calcul du vecteur orthogonal
            for i in range(j):
                projection = sum(u.data[i][k] * e.data[i][k] for k in range(n))
                u.data[j] = [u.data[j][k] - projection * e.data[i][k] for k in range(n)]
            
            # Normalisation du vecteur orthogonal
            # norm = math.sqrt(sum(val ** 2 for val in u.data[j]))
            norm = sqrt(sum(u**2))
            if norm == 0:
                raise ValueError("Cannot perform Gram-Schmidt process: vectors are linearly dependent")
            e.data[j] = [val / norm for val in u.data[j]]
        
        return e
    
    def svd(self):
        m, n = self.shape
        
        # Calcul de A * A^T et A^T * A
        ata = self.dot(self.transpose())
        aat = self.transpose().dot(self)
        
        # Calcul des valeurs propres et des vecteurs propres
        eig_vals_ata, eig_vecs_ata = ata.eig()
        eig_vals_aat, eig_vecs_aat = aat.eig()
        
        # Tri des valeurs propres et des vecteurs propres par ordre décroissant
        sorted_indices_ata = sorted(range(len(eig_vals_ata)), key=lambda i: eig_vals_ata[i], reverse=True)
        sorted_indices_aat = sorted(range(len(eig_vals_aat)), key=lambda i: eig_vals_aat[i], reverse=True)
        
        # Calcul des valeurs singulières et des vecteurs singuliers
        singular_values = [math.sqrt(eig_vals_ata[i]) for i in sorted_indices_ata if eig_vals_ata[i] > 0]
        singular_vectors = [eig_vecs_aat[i] for i in sorted_indices_aat if eig_vals_aat[i] > 0]
        
        # Remplissage des vecteurs singuliers manquants pour obtenir la taille correcte
        num_singular_vectors = min(m, n)
        if len(singular_vectors) < num_singular_vectors:
            singular_vectors.extend([Array([[0] * m] * n) for _ in range(num_singular_vectors - len(singular_vectors))])
        
        return singular_values, singular_vectors


    def eig(self):
        if not Array.is_square_matrix(self):
            raise ValueError("Cannot compute eigenvalues and eigenvectors: array is not square")

        n = self.shape[0]
        # Initialize eigenvalues and eigenvectors
        eigenvalues = []
        eigenvectors = []

        # Convert the array to a nested list for easier manipulation
        arr_list = self.data

        # Iteratively find eigenvalues and eigenvectors
        for i in range(n):
            # Initialize a random unit vector as the initial guess for the eigenvector
            v = Array.random((n, 1))
            v_prev = Array.zeros((n, 1))
            
            # Iterate until convergence
            while not Array.array_equal(v, v_prev):
                v_prev = v
                # Compute Av
                Av = self.dot(v)
                # Compute the norm of Av
                # norm_Av = math.sqrt(sum(val ** 2 for val in Av.data))
                norm_Av = math.sqrt((Av**2).sum())
                # Normalize Av
                v = Av/norm_Av
            
            # Compute the corresponding eigenvalue
            eigenvalue = v.transpose().dot(self).dot(v).data[0][0]
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)

            # Update the matrix by subtracting the contribution of this eigenvalue
            arr_list = [[arr_list[i][j] - eigenvalue * v.data[i][0] * v.data[j][0] for j in range(n)] for i in range(n)]
        
        # Convert the results back to the Array class
        eigenvalues = Array(eigenvalues)
        eigenvectors = [Array(vec) for vec in eigenvectors]

        return eigenvalues, eigenvectors

    def eigenvalues(self):
        if not Array.is_square_matrix(self):
            raise ValueError("Cannot compute eigenvalues: array is not square")

        n = self.shape[0]
        eigenvalues = []
        max_iterations = 10000
        tolerance = 1e-9

        # Create a copy of the original matrix
        A = self.copy()

        # Iterate to find eigenvalues
        for _ in range(n):
            # Initialize a random unit vector
            v = Array.random((n, 1))
            v = v / v.norm()  # Normalize the initial vector

            # Iterate until convergence
            for i in range(max_iterations):
                v_new = A.dot(v)
                v_new = v_new / v_new.norm()  # Normalize the new vector

                # Check for convergence
                if (v_new - v).norm() < tolerance:
                    break

                v = v_new

            # Compute the eigenvalue
            eigenvalue = v.T.dot(A).dot(v).data[0][0]
            eigenvalues.append(eigenvalue)

            # Update the matrix by subtracting the contribution of this eigenvalue
            A = A - eigenvalue * v.dot(v.T)

        return eigenvalues



#  # Algebre lineaire
# @staticmethod
# def qr(self):
#     if not Array.is_square_matrix(self):
#         raise ValueError("QR decomposition can only be performed on square matrices.")
    
#     n = self.shape[0]
#     q = Array(self.data)
#     r = Array(self.data)
    
#     for j in range(n):
#         v = [q.data[i][j] for i in range(j, n)]
#         norm_v = math.sqrt(sum(val ** 2 for val in v))
#         if v[0] < 0:
#             norm_v = -norm_v
#         r.data[j][j] = norm_v
        
#         if norm_v != 0:
#             v[0] += norm_v
#             scale = math.sqrt(sum(val ** 2 for val in v))
#             q_column = [[v[i] / scale] for i in range(len(v))]
            
#             for k in range(j+1, n):
#                 q_column.append([-q.data[i][j] * q_column[i-j][0] for i in range(j, n)])
#                 r.data[j][k] = sum(q.data[i][k] * q_column[i-j+1][0] for i in range(j, n))
            
#             for i in range(j, n):
#                 q.data[i][j] = q_column[i-j][0]
    
#     return q, r

# @staticmethod
# def gram_schmidt(self):
#     if self.shape[0] != self.shape[1]:
#         raise ValueError("Cannot perform Gram-Schmidt process: array is not square")
    
#     n = self.shape[0]
#     u = Array(self.data)  # Copie des vecteurs d'origine
#     e = Array([[0] * n for _ in range(n)])  # Matrice des vecteurs orthogonaux
    
#     for j in range(n):
#         # Calcul du vecteur orthogonal
#         for i in range(j):
#             projection = sum(u.data[i][k] * e.data[i][k] for k in range(n))
#             u.data[j] = [u.data[j][k] - projection * e.data[i][k] for k in range(n)]
        
#         # Normalisation du vecteur orthogonal
#         norm = math.sqrt(sum(val ** 2 for val in u.data[j]))
#         if norm == 0:
#             raise ValueError("Cannot perform Gram-Schmidt process: vectors are linearly dependent")
#         e.data[j] = [val / norm for val in u.data[j]]
    
#     return e

# @staticmethod
# def svd(self):
#     m, n = self.shape
    
#     # Calcul de A * A^T et A^T * A
#     ata = self.dot(self.transpose())
#     aat = self.transpose().dot(self)
    
#     # Calcul des valeurs propres et des vecteurs propres
#     eig_vals_ata, eig_vecs_ata = ata.eig()
#     eig_vals_aat, eig_vecs_aat = aat.eig()
    
#     # Tri des valeurs propres et des vecteurs propres par ordre décroissant
#     sorted_indices_ata = sorted(range(len(eig_vals_ata)), key=lambda i: eig_vals_ata[i], reverse=True)
#     sorted_indices_aat = sorted(range(len(eig_vals_aat)), key=lambda i: eig_vals_aat[i], reverse=True)
    
#     # Calcul des valeurs singulières et des vecteurs singuliers
#     singular_values = [math.sqrt(eig_vals_ata[i]) for i in sorted_indices_ata if eig_vals_ata[i] > 0]
#     singular_vectors = [eig_vecs_aat[i] for i in sorted_indices_aat if eig_vals_aat[i] > 0]
    
#     # Remplissage des vecteurs singuliers manquants pour obtenir la taille correcte
#     num_singular_vectors = min(m, n)
#     if len(singular_vectors) < num_singular_vectors:
#         singular_vectors.extend([Array([[0] * m] * n) for _ in range(num_singular_vectors - len(singular_vectors))])
    
#     return singular_values, singular_vectors

# @staticmethod
# def eig(self):
#         if self.shape[0] != self.shape[1]:
#             raise ValueError("Cannot compute eigenvalues and eigenvectors: array is not square")

#         n = self.shape[0]
#         # Initialisation des valeurs propres et des vecteurs propres
#         eigenvalues = []
#         eigenvectors = []

#         # Algorithme itératif pour trouver les valeurs propres et les vecteurs propres
#         for i in range(n):
#             # Initialisation d'un vecteur propre aléatoire
#             v = Array.random((n, 1))
#             v_prev = Array([[0] for _ in range(n)])
            
#             # Itération jusqu'à convergence
#             while not v.isclose(v_prev):
#                 v_prev = v
#                 # Calcul de Av
#                 Av = self.dot(v)
#                 # Calcul de la norme de Av
#                 norm_Av = sqrt((Av**2).sum())
#                 # Normalisation de Av
#                 v = Av/norm_Av
            
#             # Calcul de la valeur propre correspondante
#             eigenvalue = v.T.dot(self).dot(v).data[0][0]
#             eigenvalues.append(eigenvalue)
#             eigenvectors.append(v)

#             # Soustraction de la contribution de cette valeur propre
#             self = self - Array([[eigenvalue * v.data[i][0] * v.data[j][0] for j in range(n)] for i in range(n)])
        
#         return eigenvalues, eigenvectors
    
# @staticmethod
# def eigenvalues(self):
#         if self.shape[0] != self.shape[1]:
#             raise ValueError("Cannot compute eigenvalues: array is not square")

#         n = self.shape[0]
#         # Initialisation de la liste des valeurs propres
#         eigenvalues = []

#         # Algorithme itératif pour trouver les valeurs propres
#         for i in range(n):
#             # Initialisation d'un vecteur propre aléatoire
#             v = Array.random((n, 1))
#             v_prev = Array([[0] for _ in range(n)])
            
#             # Itération jusqu'à convergence
#             while not v.isclose(v_prev):
#                 v_prev = v
#                 # Calcul de Av
#                 Av = self.dot(v)
#                 # Calcul de la norme de Av
#                 norm_Av = math.sqrt(sum(val ** 2 for val in Av.data))
#                 # Normalisation de Av
#                 v = Array([[val / norm_Av] for val in Av.data])
            
#             # Calcul de la valeur propre correspondante
#             eigenvalue = v.transpose().dot(self).dot(v).data[0][0]
#             eigenvalues.append(eigenvalue)

#             # Soustraction de la contribution de cette valeur propre
#             self = self - Array([[eigenvalue * v.data[i][0] * v.data[j][0] for j in range(n)] for i in range(n)])
        
#         return eigenvalues







#<<<<<<<<<< Math Functions <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<      
@staticmethod
def isarray(arr):
    if isinstance(arr, Array):
        return 1
    
    raise TypeError("Object is not a Numeric Array")

@staticmethod
def exp(arr):
        return Array([math.exp(x) for x in arr.data])

@staticmethod
def log10(arr):
    return Array([math.log10(x) for x in arr.data])

@staticmethod
def acos(arr):
        return Array([math.acos(x) for x in arr.data])

@staticmethod
def atan(arr):
        return Array([math.atan(x) for x in arr.data])

@staticmethod
def asin(arr):
        return Array([math.asin(x) for x in arr.data])

@staticmethod
def ceil(arr):
        return Array([math.ceil(x) for x in arr.data])

@staticmethod
def floor(arr):
        return Array([math.floor(x) for x in arr.data])

@staticmethod
def log2(arr):
        return Array([math.log2(x) for x in arr.data])

@staticmethod
def erf(arr):
        return Array([math.erf(x) for x in arr.data])

@staticmethod
def asin(arr):
        return Array([math.asin(x) for x in arr.data])

@staticmethod
def pow(arr):
        return Array([math.pow(x) for x in arr.data])

@staticmethod
def sqrt(arr):
        return Array([math.sqrt(x) for x in arr.data])

@staticmethod
def log(arr):
        return Array([math.log(x) for x in arr.data])

@staticmethod
def sin(arr):
        return Array([math.sin(x) for x in arr.data])

@staticmethod
def cos(arr):
       return Array([math.cos(x) for x in arr.data])



#::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

