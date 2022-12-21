import os
import sys
from itertools import permutations


def permut(input_path: str) -> list:
    filelist = os.listdir(input_path)
    result = list(permutations(filelist, 2))
    return result


if __name__ == "__main__":
    sys.stdout = open("test_pairs.txt", "w")
    filelist = permut(input_path="./image")
    for file in filelist:
        cloth, human = file
        print(cloth, human)
    sys.stdout.close()
