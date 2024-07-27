import sys
from config import *
from msa_solver import CVX_ADMM_MSA

def parse_seqs_file(fname):
    with open(fname, 'r') as seq_file:
        allSeqs = []
        numSeq = 0
        for tmp_str in seq_file:
            tmp_str = tmp_str.strip()
            ht_tmp_seq = ['*'] + list(tmp_str) + ['#']
            allSeqs.append(ht_tmp_seq)
            numSeq += 1
    return allSeqs, numSeq

def get_init_model_length(lenSeqs):
    return max(lenSeqs)

def sequence_dump(allSeqs):
    for seq in allSeqs:
        print(''.join(seq))

def main():
    args = sys.argv[1:]
    fname = args[0]
    allSeqs, numSeq = parse_seqs_file(fname)
    lenSeqs = [len(seq) for seq in allSeqs]
    T2 = get_init_model_length (lenSeqs) + LENGTH_OFFSET
    sequence_dump(allSeqs)
    W = CVX_ADMM_MSA(allSeqs, lenSeqs, T2)

if __name__ == "__main__":
    main()