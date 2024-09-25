import pytest
from matrix import Matrix, Vector


def test_init_repr_shape():
    m_data = Matrix([[1, 2.5], [3, 4.125], [27, 42]])
    m_shape = Matrix((3, 2))
    print(m_data)
    print(m_shape)
    assert (
        repr(m_data)
        == str(m_data)
        == "Matrix([[1.0, 2.5], [3.0, 4.125], [27.0, 42.0]])"
    )
    assert (
        repr(m_shape) == str(m_shape) == "Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])"
    )
    assert m_data.shape == (3, 2)
    assert m_shape.shape == (3, 2)
    with pytest.raises(Exception):
        Matrix(3)
    with pytest.raises(Exception):
        Matrix([])
    with pytest.raises(Exception):
        Matrix([[], []])
    with pytest.raises(Exception):
        Matrix(3, 2)
    with pytest.raises(Exception):
        Matrix([27, 42])
    with pytest.raises(Exception):
        Matrix([[1, 2], [3, 4, 5]])
    with pytest.raises(Exception):
        Matrix([[1j, 2j], [3j, 4j]])


def test_add_sub():
    m1 = Matrix([[1, 2], [3, 4], [5, 6]])
    m2 = Matrix([[7, 8], [9, 10], [11, 42]])
    madd = Matrix([[8, 10], [12, 14], [16, 48]])
    msub12 = Matrix([[-6, -6], [-6, -6], [-6, -36]])
    msub21 = Matrix([[6, 6], [6, 6], [6, 36]])
    assert m1 + m2 == madd
    assert m1.__add__(m2) == madd
    assert m1.__radd__(m2) == madd
    assert m2 + m1 == madd
    assert m2.__add__(m1) == madd
    assert m2.__radd__(m1) == madd
    assert m1 - m2 == msub12
    assert m1.__sub__(m2) == msub12
    assert m1.__rsub__(m2) == msub21
    assert m2 - m1 == msub21
    assert m2.__sub__(m1) == msub21
    assert m2.__rsub__(m1) == msub12
    with pytest.raises(Exception):
        Matrix([[1, 2], [3, 4], [5, 6]]) + Matrix([[7, 8, 9], [10, 11, 12]])
    with pytest.raises(Exception):
        Matrix([[1, 2], [3, 4], [5, 6]]) - Matrix([[7, 8, 9], [10, 11, 12]])
    with pytest.raises(Exception):
        Matrix([[1, 2], [3, 4], [5, 6]]) + 42
    with pytest.raises(Exception):
        Matrix([[1, 2], [3, 4], [5, 6]]) - 42
    with pytest.raises(Exception):
        42 + Matrix([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(Exception):
        42 - Matrix([[1, 2], [3, 4], [5, 6]])


def test_div():
    m = Matrix([[1, 2], [3, 4], [5, 42]])
    m_quarter = Matrix([[0.25, 0.5], [0.75, 1], [1.25, 10.5]])
    assert m / 4 == m_quarter
    with pytest.raises(Exception):
        m / 0
    with pytest.raises(Exception):
        4 / m


def test_vector():
    v_data = Vector([[1, 2, 3]])
    v_shape = Vector((3, 1))
    print(v_data)
    print(v_shape)
    assert repr(v_data) == str(v_data) == "Vector([[1.0, 2.0, 3.0]])"
    assert repr(v_shape) == str(v_shape) == "Vector([[0.0], [0.0], [0.0]])"
    assert v_data.shape == (1, 3)
    assert v_shape.shape == (3, 1)
    with pytest.raises(Exception):
        Vector([[1, 2.5], [3, 4.125], [27, 42]])
    with pytest.raises(Exception):
        Vector((3, 2))
    with pytest.raises(Exception):
        Vector(3)
    with pytest.raises(Exception):
        Vector([])
    with pytest.raises(Exception):
        Vector([[], []])
    with pytest.raises(Exception):
        Vector(3, 2)
    with pytest.raises(Exception):
        Vector([27, 42])
    with pytest.raises(Exception):
        Vector([[1j, 2j]])
    assert Vector([[1, 2, 3]]) != Matrix([[1, 2, 3]])


def test_transpose_dot():
    assert Matrix([[1, 2], [3, 4]]).T() == Matrix([[1, 3], [2, 4]])
    assert Matrix([[1, 2], [3, 4], [5, 42]]).T() == Matrix([[1, 3, 5], [2, 4, 42]])
    v_row = Vector([[1, 2, 3]])
    v_col = Vector([[1], [2], [3]])
    assert v_row.T() == v_col
    assert v_col.T() == v_row
    assert len(v_row) == len(v_col) == 3
    assert (
        v_row.dot(v_col)
        == v_col.dot(v_row)
        == v_row.dot(v_row)
        == v_col.dot(v_col)
        == 14
    )
    with pytest.raises(Exception):
        v_row.dot(Vector([[1, 2, 3, 4, 5]]))


def test_mul():
    m22 = Matrix([[0, 1], [2, 3]])
    mi2 = Matrix([[1, 0], [0, 1]])
    assert m22 * mi2 == m22 * mi2 == m22
    m32 = Matrix([[4, 0], [6, 6], [6, 7]])
    assert m32 * m22 == Matrix([[0, 4], [12, 24], [14, 27]])
    m23 = Matrix([[1, 2, 3], [5, 7, 9]])
    assert m23 * m32 == Matrix([[34, 33], [116, 105]])
    assert m32 * m23 == Matrix([[4, 8, 12], [36, 54, 72], [41, 61, 81]])
    assert m23 * 3 == Matrix([[3, 6, 9], [15, 21, 27]])
    with pytest.raises(Exception):
        m22 * m32
    with pytest.raises(Exception):
        m22 * "xd"
    v13 = Vector([[1, 2, 3]])
    v31 = Vector([[5], [7], [8]])
    assert v13 * v31 == Vector([[v13.dot(v31)]])
    assert v13 * m32 == Vector([[34, 33]])
    assert m23 * v31 == Vector([[43], [146]])
    with pytest.raises(Exception):
        v13 * m23
    assert v13 * 1.5 == Vector([[1.5, 3, 4.5]])
    assert v31 * 1.5 == Vector([[7.5], [10.5], [12]])


def test_subject():
    def test1():
        m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        assert m1.shape == (3, 2)
        assert m1.T() == Matrix([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
        assert m1.T().shape == (2, 3)

    def test2():
        m1 = Matrix([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
        m1.shape == (2, 3)
        m1.T() == Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        m1.T().shape == (3, 2)

    def test3():
        m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
        m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
        assert m1 * m2 == Matrix([[28.0, 34.0], [56.0, 68.0]])

    def test4():
        m1 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
        v1 = Vector([[1], [2], [3]])
        assert m1 * v1 == Vector([[8], [16]])

    def test5():
        v1 = Vector([[1], [2], [3]])
        v2 = Vector([[2], [4], [8]])
        assert v1 + v2 == Vector([[3], [6], [11]])

    test1()
    test2()
    test3()
    test4()
    test5()
