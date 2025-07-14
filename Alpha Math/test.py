class Array:
    def __init__(self, data):
        self.data = data
        self.shape = self._get_shape(data)
        self.ndim = len(self.shape)

    @property
    def T(self):
        return self.transpose()

    def transpose(self, axes=None):
        if axes is None:
            axes = list(range(self.ndim))[::-1]
        if len(axes) != self.ndim:
            raise ValueError("Axes length must match array dimensions")
        
        transposed_data = self._transpose_recursive(self.data, axes)
        return Array(transposed_data)

    def _transpose_recursive(self, data, axes):
        if len(axes) == 1:
            return data
        current_axis = axes[0]
        remaining_axes = axes[1:]
        
        if isinstance(data[0], list):
            transposed = list(map(list, zip(*data)))
            transposed = [self._transpose_recursive(sublist, remaining_axes) for sublist in transposed]
            return transposed
        else:
            return data

    def _reorder_axes(self, data, order):
        if not order:
            return data
        if isinstance(data[0], list):
            return list(map(list, zip(*[self._reorder_axes(sublist, order[1:]) for sublist in data])))
        else:
            return data

    def _get_shape(self, data):
        if isinstance(data, list):
            if all(isinstance(i, list) for i in data):
                sub_shape = self._get_shape(data[0])
                return (len(data),) + sub_shape
            else:
                return (len(data),)
        else:
            return ()

# Example usage:
arr = Array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr.T.data)  # Regular transpose

arr_3d = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(arr_3d.transpose().data)  # Transpose for 3D array
print(arr_3d.transpose((9, 0, 1,)).data)  # Transpose according to shape (1, 0, 2)
