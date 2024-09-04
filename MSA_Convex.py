import sys, os
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
    solver = CVX_ADMM_MSA(allSeqs, lenSeqs, T2)
    
    print(">>>>>>>>>>>>>>>>>>>>>>>SequenceView<<<<<<<<<<<<<<<<<<<<<<<<")
    recSeq = solver.recSeq
    #recSeq = "*GCTGTGCATTCGCGGCACAAGAGTCCCGGG#"
    for n,seq in enumerate(allSeqs):
        print("{:<10}".format("Seq "+str(n)+":"), ''.join(seq))
    print("{:<10}".format("SeqRecov:"), ''.join(recSeq))
    
    print(">>>>>>>>>>>>>>>>>>>>>>>MatchingView<<<<<<<<<<<<<<<<<<<<<<<<")
    allModelSeqs, allDataSeqs = [], []
    model_seq = recSeq[1:-1]
    for data_seq in allSeqs:
        data_seq = data_seq[1:-1]
        trace = CVX_ADMM_MSA.smith_waterman(model_seq, data_seq)
        model_seq = [t.acidA for t in trace]
        data_seq = [t.acidB for t in trace]
        allModelSeqs.append(model_seq)
        allDataSeqs.append(data_seq)
        print(''.join(model_seq))
        print(''.join(data_seq))

    print(">>>>>>>>>>>>>>>>>>>>>ClustalOmegaView<<<<<<<<<<<<<<<<<<<<<<")
    allCOSeqs = [[] for _ in range(numSeq)]
    pos = [0 for _ in range(numSeq)]
    while True:
        insertion_ids = set()
        for i in range(numSeq):
            if pos[i] >= len(allModelSeqs[i]):
                continue
            if allModelSeqs[i][pos[i]] == '-':
                insertion_ids.add(i)

        if insertion_ids: # Insertion exists
            for i in range(numSeq):
                if i not in insertion_ids:
                    allCOSeqs[i].append('-')
                else:
                    allCOSeqs[i].append(allDataSeqs[i][pos[i]])
                    pos[i] += 1
        else: # No insertion
            for i in range(numSeq):
                allCOSeqs[i].append(allDataSeqs[i][pos[i]])
                pos[i] += 1
        
        if all([pos[i] == len(allModelSeqs[i]) for i in range(numSeq)]):
            break
    
    if WRITEFILE:
        fname = os.path.splitext(os.path.basename(fname))[0]
        f = open(fname+".co", "w")
    for seq in allCOSeqs:
        print(''.join(seq))
        if WRITEFILE:
            f.write(''.join(seq)+'\n')
    if WRITEFILE:
        f.close()

    print("#########################################################")
    print("end")


if __name__ == "__main__":
    main()