import pytest
import csv
import load


def test_smooth_loss():
    model = load.loadModel()
    file = open(model.dir, encoding="utf8")
    csvreader = csv.reader(file)
    loss = 10
    for row in csvreader:
        loss = row[1]
    assert float(loss) < 2.0


if __name__ == '__main__':
    test_smooth_loss()
