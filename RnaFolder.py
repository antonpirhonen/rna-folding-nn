
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import RNA

# Add the folded sequence to the dataframe using python 2.7 without pandas or other libraries

fread = open("data.csv", "r")
fwrite = open("data_with_folded.csv", "w")

# Write the header
fwrite.write("seq,folded_seq\n")

for line in fread:
    seq = line.rstrip()
    fold = RNA.fold(seq)
    fwrite.write(seq + "," + fold[0] + "\n")

fread.close()
fwrite.close()