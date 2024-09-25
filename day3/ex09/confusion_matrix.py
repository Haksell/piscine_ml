import numpy as np


def confusion_matrix_(y, y_hat, *, labels=None, df_option=False):
    if (
        isinstance(y, np.ndarray)
        and isinstance(y_hat, np.ndarray)
        and y.shape == y_hat.shape
    ):
        if labels is None:
            labels = sorted(set(np.unique(y)) | set(np.unique(y_hat)))
        labels_idx = {label: i for i, label in enumerate(labels)}
        cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
        for actual, predicted in zip(y.flatten(), y_hat.flatten()):
            if actual in labels_idx and predicted in labels_idx:
                actual_idx = labels_idx[actual]
                predicted_idx = labels_idx[predicted]
                cm[actual_idx, predicted_idx] += 1
        if df_option:
            str_grid = [[""] + labels] + [[label] for label in labels]
            for i, row in enumerate(cm):
                for val in row:
                    str_grid[i + 1].append(str(val))
            widths = [max(map(len, col)) for col in zip(*str_grid)]
            return "\n".join(
                "  ".join(cell.rjust(width) for cell, width in zip(row, widths))
                for row in str_grid
            )
        else:
            return cm


def __main():
    from sklearn.metrics import confusion_matrix

    y = np.array([["dog"], ["dog"], ["norminet"], ["norminet"], ["dog"], ["norminet"]])
    y_hat = np.array(
        [["norminet"], ["dog"], ["norminet"], ["norminet"], ["dog"], ["bird"]]
    )
    print(confusion_matrix_(y, y_hat))
    print(confusion_matrix(y, y_hat))
    print()
    print(confusion_matrix_(y, y_hat, labels=["dog", "norminet"]))
    print(confusion_matrix(y, y_hat, labels=["dog", "norminet"]))
    print()
    print(confusion_matrix_(y, y_hat, df_option=True))
    print()
    print(confusion_matrix_(y, y_hat, labels=["bird", "dog"], df_option=True))


if __name__ == "__main__":
    __main()
