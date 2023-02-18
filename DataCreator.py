import pandas as pd
import random

class DataCreator:
    def __init__(self):
        pass

    def randomNucleotide(self):
        r = random.randrange(4)
        if r == 0:
            return "A"
        if r == 1:
            return "C"
        if r == 2:
            return "G"
        if r == 3:
            return "U"
        raise Exception("Choosing a random nucleotide failed.")
        
    def createSequence(self, length):
        array = [0] * length
        i = 0
        while i < length:
            array[i] = self.randomNucleotide()
            i += 1
        return "".join(array)

    def createData(self, seq_length, n, output_file):
        data = []
        i = 0
        while i < n:
            seq = self.createSequence(seq_length)
            data.append(seq)
            i += 1
        df = pd.DataFrame(data, columns=["sequence"])
        # Store data withput header
        df.to_csv(output_file, index=False, header=False)

dc = DataCreator()
dc.createData(30, 100000, "data.csv")