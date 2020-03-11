#!/usr/bin/python3

from NaiveBayes import NB
import os

home = os.environ['HOME']

nb = NB(26, 784)

nb.train(home + "/data/letters_csv_train.dat", 124800)

nb.predict(0.05)
