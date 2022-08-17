import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def parseFile(fileIn, fileOut):
    f = open(fileIn, "r", encoding="utf-8")
    lines = f.readlines()

    data = []
    step = 0
    run = -1
    i = 0
    times = []

    while i < len(lines):

        line = lines[i].split()
        #print(line)

        if line != []:
            if len(line) > 2 and line[1] == "seconds":
                print(" ".join(line))
            if line[0] == "Time" and line[-1] == "seconds":
                times.append(float(line[-2]))

            if line[0] == "Progress:":
                step = line[1]
            elif (line[0] == "Activating" or line[0] == "\ufeffActivating"):
                run += 1
                print("run", run, "at line", i)
                while (i < len(lines)-1 and (line == [] or line[0] != "Progress:")):
                    #print("skipping", i)
                    i += 1
                    line = lines[i].split()
            else:
                try:
                    complexity = int(line[0])
                    data.append([run] + [step] + [times[-1]] + line[:3] + [" ".join(line[3:])])
                except:
                    # meaningless but we need to have something here
                    complexity = 0

        i += 1

    data = pd.DataFrame(data, columns=["run", "progress", "runtime", "complexity", "loss", "score", "equation"])
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    #print(data)

    data.to_csv(fileOut)

    print(times)

def main(argv):
    if len(argv) != 3:
        print("Requires 2 arguments (in and out file)")
        return
    else:
        parseFile(argv[1], argv[2])

if __name__ == "__main__":
   main(sys.argv)
