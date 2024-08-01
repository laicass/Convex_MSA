import numpy as np
from config import *
import util

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

        #util.print_C(self.C)
        self.mu = MU
        self.prev_CoZ = MAX_DOUBLE
        for iter in range(MAX_ADMM_ITER):
            # First subproblem
            for n in range(self.numSeq):
                self.first_subproblem(self.W_1[n], self.W_2[n], self.Y[n], self.C[n], self.mu, self.allSeqs[n])

            # Second subproblem
            self.second_subproblem(self.W_1, self.W_2, self.Y, self.mu, self.allSeqs, self.lenSeqs)

            # Dual upgrade
            for n in self.numSeq:
                self.tensor4D_lin_update(self.Y[n], self.W_1[n], self.W_2[n], self.mu)

    def first_subproblem(self, W_1, W_2, Y, C, mu, data_seq):
        W_1 = np.zeros_like(W_1)
        M = Y - mu * W_2
        alpha_lookup = {}
        for fw_iter in range(MAX_1st_FW_ITER):
            S_atom, trace = self.cube_smith_waterman(M, C, data_seq)
            S_atom = np.array(S_atom).reshape((-1,4))
            S_atom = tuple(map(tuple, S_atom))
            gfw_W = gfw_S = 0.0
            for key, value in alpha_lookup.items():
                gfw_W += sum([(C[idx] + M[idx]) * value for idx in key])
            
            gfw_S -= sum([(C[idx] + M[idx]) for idx in S_atom])

            gfw = gfw_S + gfw_W
            print(gfw)
            if fw_iter > 0 and gfw < FW1_GFW_EPS:
                break

            # find away direction
            V_atom = tuple(tuple())
            gamma_max = 1.0
            if len(alpha_lookup.keys()) > 0:
                V = []
                for key, value in alpha_lookup.items(): 
                    cpm = sum([(C[idx] + M[idx]) for idx in key])
                    V.append((cpm, key, value))
                max_val, V_atom, gamma_max = max(V)

            # exact line search
            numerator = denominator = 0.0
            smv_lookup = {}
            for idx in S_atom:
                numerator += mu * (W_2[idx] - W_1[idx]) - C[idx] - Y[idx]
                smv_lookup[idx] = 1.0
                denominator += mu
            for idx in V_atom:
                numerator -= mu * (W_2[idx] - W_1[idx]) - C[idx] - Y[idx]
                if idx not in smv_lookup:
                    denominator += mu
                else:
                    denominator -= mu

            # early stop
            gamma = numerator / denominator
            if REINIT_W_ZERO_TOGGLE and fw_iter == 0:
                gamma = 1.0
            gamma = min(max(gamma, 0.0), gamma_max)

            # update W_1
            for idx in map(tuple, S_atom):
                W_1[idx] += gamma
                M[idx] += mu * gamma
            for idx in map(tuple, V_atom):
                W_1[idx] -= gamma
                M[idx] -= mu * gamma

            if len(alpha_lookup) == 0:
                alpha_lookup[S_atom] = 1.0
            else:
                if S_atom in alpha_lookup:
                    alpha_lookup[S_atom] += gamma
                else:
                    alpha_lookup[S_atom] = gamma
                if alpha_lookup[V_atom] - gamma < 1e-10:
                    alpha_lookup.pop(V_atom)
                else:
                    alpha_lookup[V_atom] -= gamma

        exit(0)

    def second_subproblem(self, W_1, W_2, Y, mu, allSeqs, lenSeqs):
        pass

    def tensor4D_lin_update(self, Y, W_1, W_2, mu):
        pass

    def cube_smith_waterman(self, M, C, data_seq):
        trace = []
        T1, T2, T3 = C.shape[:3]
        cube = [[[util.Cell(3) for _ in range(T3)] for _ in range(T2)] for _ in range(T1)]
        for i in range(T1):
            for j in range(T2):
                cube[i][j][4].score = MAX_DOUBLE
        for k in range(T3):
            cube[0][0][k].score = MAX_DOUBLE
        acc_ins_cost = C[0][0][4][11] + M[0][0][4][11]

        for i in range(1, T1):
            acc_ins_cost += C[i][0][4][0] + M[i][0][4][0]
            cube[i][0][4].score = acc_ins_cost 
            cube[i][0][4].ans_idx = INSERTION 
            cube[i][0][4].action = INSERTION 
            cube[i][0][4].acidA = data_seq[i] 
            cube[i][0][4].acidB = GAP_NOTATION

        cube[0][0][4].score = C[0][0][4][11] + M[0][0][4][11]
        cube[0][0][4].ans_idx = MATCH_START
        cube[0][0][4].action = MATCH_START
        cube[0][0][4].acidA = '*'
        cube[0][0][4].acidB = '*'

        for i in range(T1):
            for j in range(T2):
                for k in range(T3):
                    cube[i][j][k].location = [i, j, k]
                    if (i == 0 and j == 0) or k == 4:
                        continue
                    data_dna = data_seq[i]
                    scores = [0.0] * NUM_MOVEMENT
                    
                    # 1a. get insertion score
                    ins_score = (MAX_DOUBLE if i == 0 else cube[i-1][j][k].score) + \
                                M[i][j][k][INS_BASE_IDX] + \
                                C[i][j][k][INS_BASE_IDX]
                    scores[INS_BASE_IDX] = ins_score
                    
                    # 1b. get deletion score
                    for d in range(NUM_DNA_TYPE):
                        del_score = (MAX_DOUBLE if j == 0 else cube[i][j-1][d].score) + \
                                    M[i][j][d][DEL_BASE_IDX + k] + \
                                    C[i][j][d][DEL_BASE_IDX + k]
                        scores[DEL_BASE_IDX + d] = del_score
                    
                    # 1c. get max match/mismatch score
                    for d in range(NUM_DNA_TYPE):
                        mth_score = (MAX_DOUBLE if i == 0 or j == 0 else cube[i-1][j-1][d].score) + \
                                    M[i][j][d][MTH_BASE_IDX + k] + \
                                    C[i][j][d][MTH_BASE_IDX + k]
                        scores[MTH_BASE_IDX + d] = mth_score

                    # 1d. get optimal action for the current cell
                    min_score = min(scores)
                    if min_score == MAX_DOUBLE:
                        min_ansid = -1
                    else:
                        min_ansid = scores.index(min_score)
                    if min_ansid == INS_BASE_IDX:
                        min_action = INSERTION
                    elif DEL_BASE_IDX <= min_ansid < MTH_BASE_IDX:
                        min_action = list(util.action2str.keys())[DEL_BASE_IDX + k]
                    elif MTH_BASE_IDX <= min_ansid < NUM_MOVEMENT:
                        min_action = list(util.action2str.keys())[MTH_BASE_IDX + k]
                    else:
                        min_action = 6
                    #print(f", min_ansid = {min_ansid}, min_score = {int(min_score)}, min_action = {min_action}")

                    # 1e. assign the optimal score/action to the cell
                    cube[i][j][k].score = min_score
                    cube[i][j][k].action = min_action
                    cube[i][j][k].ans_idx = min_ansid
                    act = cube[i][j][k].action
                    if act == INSERTION:
                        cube[i][j][k].acidA = data_dna
                        cube[i][j][k].acidB = GAP_NOTATION
                    elif DEL_BASE_IDX <= act < MTH_BASE_IDX:
                        cube[i][j][k].acidA = GAP_NOTATION
                        cube[i][j][k].acidB = util.T3idx2dna(act - DEL_BASE_IDX)
                    elif act >= MTH_BASE_IDX or act < NUM_DNA_TYPE:
                        cube[i][j][k].acidA = data_dna
                        cube[i][j][k].acidB = util.T3idx2dna(act - MTH_BASE_IDX)
                    else:
                        print("uncatched action.", act)

        # 3. Track back
        tmp = [x[END_IDX].score for x in (cube[T1-1])[1:T2]]
        global_min_score = min(tmp)
        gmin_i = T1-1
        gmin_j = -1
        gmin_k = END_IDX
        gmin_j = tmp.index(global_min_score) + 1
        
        if gmin_i == 0 or gmin_j == 0:
            trace.append(cube[gmin_i][gmin_j][gmin_k])
            return
        i = gmin_i; j = gmin_j; k = gmin_k
        while i >= 0 and j >= 0:
            trace.insert(0, cube[i][j][k])
            act = cube[i][j][k].action
            ans_idx = cube[i][j][k].ans_idx
            if act == INS_BASE_IDX:
                i-=1
            elif DEL_BASE_IDX <= act and act < MTH_BASE_IDX:
                j-=1
                k = (ans_idx - MTH_BASE_IDX) if (ans_idx >= MTH_BASE_IDX) else (ans_idx - DEL_BASE_IDX)
            elif MTH_BASE_IDX <= act and act < NUM_MOVEMENT:
                i-=1
                j-=1
                k = (ans_idx - MTH_BASE_IDX) if (ans_idx >= MTH_BASE_IDX) else (ans_idx - DEL_BASE_IDX)
            else:
                pass

        # 4. reintepret it as 4-d data structure
        S_atom = []
        for t in range(len(trace)):
            i,j,k = trace[t].location
            m = trace[t].action
            S_atom.append(i)
            S_atom.append(j)
            if t == 0:
                S_atom.append(4)
            else:
                S_atom.append(trace[t-1].location[2])
            S_atom.append(m)
        return S_atom, trace
    

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
                                C[n][i][j][k][m] = C_M if util.dna2T3idx(allSeqs[n][i]) == m - MTH_BASE_IDX else C_MM
                            #C[n][i][j][k][m] += PERB_EPS * (np.random.rand())
                            #print((n,i,j,k,m), allSeqs[n][i], C[n][i][j][k][m])
        return C
    
    


    def tensor5D_init(self, init_T2):
        tensor5D = []
        for i in range(self.numSeq):
            tmp_tensor = np.zeros((len(self.allSeqs[i]),init_T2, NUM_DNA_TYPE, NUM_MOVEMENT))
            tensor5D.append(tmp_tensor)
        return tensor5D