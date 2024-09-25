class Matrix:
    def __init__(self, data):
        if isinstance(data, tuple):
            assert len(data) == 2
            self._height = data[0]
            self._width = data[1]
            assert isinstance(self._height, int) and self._height > 0
            assert isinstance(self._width, int) and self._width > 0
            self._data = [[0.0] * self._width for _ in range(self._height)]
        else:
            assert isinstance(data, list)
            assert all(isinstance(row, list) for row in data)
            self._height = len(data)
            assert self._height > 0
            self._width = len(data[0])
            assert self._width > 0
            assert all(len(row) == len(data[0]) for row in data)
            assert all(isinstance(x, (int, float)) for row in data for x in row)
            self._data = [list(map(float, row)) for row in data]
        self._shape = (self._height, self._width)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def shape(self):
        return self._shape

    def __eq__(self, other):
        # use type() instead of isinstance() to distinguish between Matrix and Vector
        return (
            type(self) == type(other)
            and self._shape == other._shape
            and all(
                x1 == x2
                for r1, r2 in zip(self._data, other._data)
                for x1, x2 in zip(r1, r2)
            )
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self._data})"

    __str__ = __repr__  # also works implicitly

    def __add__(self, other):
        assert isinstance(other, Matrix)
        assert self._shape == other._shape
        ret_class = Matrix if type(self) == Matrix or type(other) == Matrix else Vector
        return ret_class(
            [
                [x1 + x2 for x1, x2 in zip(r1, r2)]
                for r1, r2 in zip(self._data, other._data)
            ]
        )

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        assert isinstance(other, Matrix)
        assert self._shape == other._shape
        ret_class = Matrix if type(self) == Matrix or type(other) == Matrix else Vector
        return ret_class(
            [
                [x1 - x2 for x1, x2 in zip(r1, r2)]
                for r1, r2 in zip(self._data, other._data)
            ]
        )

    def __rsub__(self, other):
        return other - self

    def __truediv__(self, scalar):
        assert isinstance(scalar, (float, int)) and scalar != 0
        return self.__class__([[x / scalar for x in row] for row in self._data])

    def __rtruediv__(self, other):
        raise TypeError(
            f"Can't divide an object of type {other.__class__.__name__} by a {self.__class__.__name__}"
        )

    def T(self):
        return self.__class__(list(map(list, zip(*self._data))))

    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.height
            ret_class = (
                Matrix if type(self) == Matrix and type(other) == Matrix else Vector
            )
            return ret_class(
                [
                    [sum(x * y for x, y in zip(row, col)) for col in other.T()._data]
                    for row in self._data
                ]
            )
        else:
            assert isinstance(other, (float, int))
            return self.__class__([[x * other for x in row] for row in self._data])

    def __rmul__(self, other):
        return other * self


class Vector(Matrix):
    def __init__(self, data):
        super().__init__(data)
        assert self.width == 1 or self.height == 1

    def __len__(self):
        return self.width * self.height

    def dot(self, other):
        assert isinstance(other, Vector)
        assert len(self) == len(other)
        return sum(x * y for x, y in zip(self, other))

    def __iter__(self):
        if self.height == 1:
            yield from self._data[0]
        else:
            for row in self._data:
                yield row[0]
