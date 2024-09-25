from data_spliter import data_spliter
import numpy as np

x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
x2 = np.array([[1, 42], [300, 10], [59, 1], [300, 59], [10, 42]])
y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))


def test_data_spliter_size():
    assert [a.shape for a in data_spliter(x1, y, 0.8)] == [
        (4, 1),
        (1, 1),
        (4, 1),
        (1, 1),
    ]
    assert [a.shape for a in data_spliter(x1, y, 0.5)] == [
        (2, 1),
        (3, 1),
        (2, 1),
        (3, 1),
    ]
    assert [a.shape for a in data_spliter(x2, y, 0.8)] == [
        (4, 2),
        (1, 2),
        (4, 1),
        (1, 1),
    ]
    assert [a.shape for a in data_spliter(x2, y, 0.5)] == [
        (2, 2),
        (3, 2),
        (2, 1),
        (3, 1),
    ]


def test_data_spliter_full():
    def check_full(x, y, p):
        x_train, x_test, y_train, y_test = data_spliter(x, y, p)
        assert sorted(map(list, np.vstack([x_train, x_test]))) == sorted(map(list, x))
        assert sorted(map(list, np.vstack([y_train, y_test]))) == sorted(map(list, y))

    check_full(x1, y, 0.8)
    check_full(x1, y, 0.5)
    check_full(x2, y, 0.8)
    check_full(x2, y, 0.5)
