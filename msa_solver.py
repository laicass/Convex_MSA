import numpy as np
from config import *

class CVX_ADMM_MSA:
    def __init__(self, allSeqs, lenSeqs, T2):
        self.allSeqs = allSeqs
        self.lenSeqs = lenSeqs
        self.numSeq = len(allSeqs)
        self.T2 = T2
        
        self.C = self.tensor5D_init(self.T2)
        self.W_1 = self.tensor5D_init(self.T2)
        self.W_2 = self.tensor5D_init(self.T2)
        self.Y = self.tensor5D_init(self.T2)
        self.C = self.set_C(self.C, self.allSeqs)
        self.print_C(self.C)

    def print_C(self, C):
        T0 = len(C)
        for n in range(T0):
            T1, T2, T3, T4 = C[n].shape
            for i in range(T1):
                for j in range(T2):
                    for k in range(T3):
                        for m in range(T4):
                            print(f"C[{n}][{i}][{j}][{k}][{m}] = {int(C[n][i][j][k][m])}")

    def set_C(self, C, allSeqs):
        T0 = len(C)
        T2 = C[0][0].shape[0]
        T3 = NUM_DNA_TYPE
        T4 = NUM_MOVEMENT
        for n in range(T0):
            T1 = len(allSeqs[n])  # length of each sequence
            for i in range(T1):
                for j in range(T2):
                    for k in range(T3):
                        for m in range(T4):
                            if m == INS_BASE_IDX:
                                if allSeqs[n][i] == '#':
                                    C[n][i][j][k][m] = HIGH_COST
                                    #print((n,i,j,k,m), C[n][i][j][k][m])
                                    continue
                                C[n][i][j][k][m] = C_I
                            # DELETION PENALTIES
                            elif m == DELETION_START:  # disallow delete *
                                C[n][i][j][k][m] = HIGH_COST # HIGH_COST
                            elif m == DELETION_END:  # disallow delete #
                                C[n][i][j][k][m] = HIGH_COST
                            elif DEL_BASE_IDX <= m < MTH_BASE_IDX:
                                C[n][i][j][k][m] = C_D
                            # MATCH PENALTIES
                            elif m == MATCH_START:
                                C[n][i][j][k][m] = NO_COST if allSeqs[n][i] == '*' else HIGH_COST  # disallow mismatch *
                            elif m == MATCH_END:
                                C[n][i][j][k][m] = NO_COST if allSeqs[n][i] == '#' else HIGH_COST  # disallow mismatch #
                            elif MTH_BASE_IDX <= m:
                                if allSeqs[n][i] == '#' and m != MATCH_END:
                                    C[n][i][j][k][m] = HIGH_COST
                                    #print((n,i,j,k,m), C[n][i][j][k][m])
                                    continue
                                C[n][i][j][k][m] = C_M if self.dna2T3idx(allSeqs[n][i]) == m - MTH_BASE_IDX else C_MM
                            #C[n][i][j][k][m] += PERB_EPS * (np.random.rand())
                            #print((n,i,j,k,m), allSeqs[n][i], C[n][i][j][k][m])
        
        return C
    
    def dna2T3idx(self, dna):
        m = {'A': 0, 'T': 1, 'C': 2, 'G': 3, '*': 4, '#': 5, GAP_NOTATION: -1}
        if dna in m:
            return m[dna]
        else:
            print("error", dna)
            exit()

    def tensor5D_init(self, init_T2):
        tensor5D = []
        for i in range(self.numSeq):
            tmp_tensor = np.zeros((len(self.allSeqs[i]),init_T2, NUM_DNA_TYPE, NUM_MOVEMENT))
            tensor5D.append(tmp_tensor)
        return tensor5D