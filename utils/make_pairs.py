import itertools
import os
import sys


def MakePairs(dataroot: str):
    fileList = os.listdir(dataroot)
    humanImgs = []
    for file in fileList:
        if "._human" not in file:
            humanImgs.append(file)

    result = list(itertools.permutations(humanImgs, 2))
    return result

if __name__ == "__main__":
    sys.stdout = open('test_pairs.txt', 'w')
    it = MakePairs('../test_samples/image')
    for i in it:
        out1, out2 = i
        print(out1, out2)
    sys.stdout.close()
