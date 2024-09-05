import numpy as np
from config import *
#from dynamic_programming import cube_smith_waterman_GPU
import util

def smith_waterman(seqA, seqB):
    plane = [[util.Cell(2) for _ in range(len(seqA)+1)] for _ in range(len(seqB)+1)]
    trace = []
    nRow = len(plane)
    nCol = len(plane[0])
    for i in range(nRow):
        for j in range(nCol):
            plane[i][j].location[0] = i
            plane[i][j].location[1] = j
            if i == 0 and j == 0:
                continue
            if i == 0 or j == 0:
                plane[i][j].score = i * C_I + j * C_D
                plane[i][j].action = INSERTION if i > 0 else DELETION_A
                plane[i][j].acidA = GAP_NOTATION if i > 0 else seqA[j-1]
                plane[i][j].acidB = GAP_NOTATION if j > 0 else seqB[i-1]
                continue

            acidA = seqA[j-1]
            acidB = seqB[i-1]
            mscore = C_M if acidA == acidB else C_MM
            mm_score = plane[i-1][j-1].score + mscore

            del_score = MAX_DOUBLE
            l = 1
            while j - l > 0:
                del_score = min(del_score, plane[i][j-l].score + l * C_D)
                l+=1

            ins_score = MAX_DOUBLE
            l = 1
            while i - l > 0:
                ins_score = min(ins_score, plane[i-l][j].score + l * C_I)
                l+=1

            opt_score = MAX_DOUBLE
            if ins_score <= min(mm_score, del_score):
                opt_score = ins_score
                opt_action = INSERTION
                opt_acidA = GAP_NOTATION
                opt_acidB = acidB
            elif del_score <= min(mm_score, ins_score):
                opt_score = del_score
                opt_action = DELETION_A
                opt_acidA = acidA
                opt_acidB = GAP_NOTATION
            elif mm_score <= min(ins_score, del_score):
                opt_score = mm_score
                opt_action = MATCH_A if acidA==acidB else MATCH_G
                opt_acidA = acidA
                opt_acidB = acidB

            plane[i][j].score = opt_score
            plane[i][j].action = opt_action
            plane[i][j].acidA = opt_acidA
            plane[i][j].acidB = opt_acidB
    
    # Finding minimum score in the last column
    min_score = MAX_DOUBLE
    min_i, min_j = -1, -1
    for i in range(nRow):
        score = plane[i][-1].score + (nRow - i - 1) * C_I
        if score <= min_score:
            min_score = score
            min_i = i
            min_j = nCol - 1
    
    # Finding minimum score in the last row
    for j in range(nCol):
        score = plane[-1][j].score + (nCol - j - 1) * C_D
        if score <= min_score:
            min_score = score
            min_i = nRow - 1
            min_j = j
    #print("min_i", min_i, "min_j", min_j)

    # Trace back
    if min_i == 0 or min_j == 0:
        trace.append(plane[min_i][min_j])
        return trace
    
    i, j = min_i, min_j
    while i != 0 or j != 0:
        trace.append(plane[i][j])
        action = plane[i][j].action
        if action == MATCH_A or action == MATCH_G:
            i -= 1
            j -= 1
        elif action == INSERTION:
            i -= 1
        elif action == DELETION_A:
            j -= 1
        else:
            raise ValueError("Uncaught action.")
    trace = trace[::-1]
    return trace

def cube_smith_waterman(M, C, data_seq):
    #print("cpu")
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
                #print(f"min_ansid = {min_ansid}, min_score = {min_score}, min_action = {min_action}")
                
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
                
                #print(act, util.dna2T3idx(cube[i][j][k].acidA), util.dna2T3idx(cube[i][j][k].acidB))
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

def refined_viterbi_algo(transition, mat_insertion):
    J, D1, D2 = transition.shape
    plane = [[util.Cell(2) for _ in range(D2)] for _ in range(J+1)]
    for j in range(J):
        t = np.array([c.score for c in plane[j]]) + mat_insertion[j] # size = D1
        score_map = (transition[j].T + t)[:,:-1]
        max_score = np.max(score_map, axis=1)
        max_d1 = np.argmax(score_map, axis=1)
    
        jp = j + 1
        for d2 in range(D2):
            jp = j + 1
            plane[jp][d2].location[0] = j
            plane[jp][d2].location[1] = max_d1[d2]
            plane[jp][d2].score = max_score[d2]
            plane[jp][d2].acidA = util.T3idx2dna(max_d1[d2])
            plane[jp][d2].acidB = util.T3idx2dna(d2)
    
    max_score, max_end_pos = max([(plane[j][END_IDX].score, j) for j in range(1,J+1)])

    trace = []
    trace.append(plane[max_end_pos][END_IDX])
    for j in range(max_end_pos-1, 0, -1):
        last_d2 = util.dna2T3idx(trace[-1].acidA)
        trace.append(plane[j][last_d2])
    trace = trace[::-1]
    return trace

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

        self.mu = MU
        self.prev_CoZ = MAX_DOUBLE
        self.recSeq = None
        for iter in range(MAX_ADMM_ITER):
            # First subproblem
            for n in range(self.numSeq):
                self.first_subproblem(self.W_1[n], self.W_2[n], self.Y[n], self.C[n], self.mu, self.allSeqs[n])

            # Second subproblem
            recSeq = self.second_subproblem(self.W_1, self.W_2, self.Y, self.mu, self.allSeqs, self.lenSeqs)
            print(recSeq)
            self.recSeq = recSeq

            # Dual upgrade
            for n in range(self.numSeq):
                self.Y[n] += self.mu * (self.W_1[n] - self.W_2[n])
            CoZ = sum([util.Frobenius_prod(self.C[n], self.W_2[n]) for n in range(self.numSeq)])
            W1mW2 = max([np.max(np.abs(self.W_1[n]-self.W_2[n])) for n in range(self.numSeq)])
            print("CoZ:", CoZ, "W1mW2:", W1mW2)
            print("ADMM iter =", iter)
            for n in range(self.numSeq):
                model_seq = recSeq[1:-1]
                data_seq = self.allSeqs[n][1:-1]

            # 2.f Stopping ADMM
            if ADMM_EARLY_STOP_TOGGLE and iter > MIN_ADMM_ITER:
                if W1mW2 < EPS_Wdiff:
                    print("CoZ Converges. ADMM early stop!")
                    break
            prev_CoZ = CoZ

    def first_subproblem(self, W_1, W_2, Y, C, mu, data_seq):
        W_1.fill(0)
        M = Y - mu * W_2
        alpha_lookup = {}
        for fw_iter in range(MAX_1st_FW_ITER+1):
            S_atom, trace = cube_smith_waterman(M, C, data_seq)
            #S_atom, trace = cube_smith_waterman_GPU(M, C, data_seq)
            S_atom = np.array(S_atom).reshape((-1,4))
            S_atom = tuple(map(tuple, S_atom))
            gfw_W = gfw_S = 0.0
            for key, value in alpha_lookup.items():
                gfw_W += sum([(C[idx] + M[idx]) * value for idx in key])
            
            gfw_S -= sum([(C[idx] + M[idx]) for idx in S_atom])

            gfw = gfw_S + gfw_W
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

    def second_subproblem(self, W_1, W_2, Y, mu, allSeqs, lenSeqs):
        numSeq = len(allSeqs)
        T2 = W_2[0][0].shape[0]
        delta = self.tensor5D_init(T2)
        tensor = np.zeros((T2, NUM_DNA_TYPE, NUM_DNA_TYPE))
        mat_insertion = np.zeros((T2, NUM_DNA_TYPE))
        for n in range(numSeq):
            W_2[n].fill(0)
            delta[n] = mu * W_1[n] + Y[n]
            i_delta = np.clip(delta[n][:,:,:,INSERTION], a_min=0.0, a_max=None)
            mat_insertion += np.sum(i_delta, axis=0)
            for m in range(INSERTION+1, NUM_MOVEMENT):
                i_delta = np.clip(delta[n][:,:,:,m], a_min=0.0, a_max=None)
                tensor[:,:,util.move2T3idx(m)] += np.sum(i_delta, axis=0)
        alpha_lookup = {}

        for fw_iter in range(MAX_2nd_FW_ITER+1):
            trace = refined_viterbi_algo(tensor, mat_insertion)
            S_atom = []
            for t in trace:
                sj = t.location[0]
                sd = t.location[1]
                sm = util.dna2T3idx(t.acidB)
                if t.acidA == '#':
                    break
                for n in range(numSeq):
                    for i in range(delta[n].shape[0]):
                        for m in range(NUM_MOVEMENT):
                            if delta[n][i][sj][sd][m] > 0.0:
                                added = False
                                if m == INSERTION or m == DEL_BASE_IDX + sm or m == MTH_BASE_IDX + sm:
                                    added = True
                                if added:
                                    S_atom.append(n);S_atom.append(i);S_atom.append(sj);S_atom.append(sd);S_atom.append(m)
            
            # Early stopping
            S_atom = np.array(S_atom).reshape((-1,5))
            S_atom = tuple(map(tuple, S_atom))
            gfw_S = gfw_W = gfw = 0.0
            gfw_S += sum([delta[idx[0]][idx[1:]] for idx in S_atom])

            for key, value in alpha_lookup.items():
                gfw_W -= sum([delta[idx[0]][idx[1:]] * value for idx in key])
            gfw = gfw_S + gfw_W
            print("gfw_S=", gfw_S, "gfw_W=", gfw_W, "gfw=", gfw)

            if fw_iter > 0 and gfw < FW2_GFW_EPS:
                print("break")
                break

            # Away step
            V_atom = tuple(tuple())
            gamma_max = 1.0
            if len(alpha_lookup.keys()) > 0:
                V = []
                for key, value in alpha_lookup.items(): 
                    sum_delta = sum([delta[idx[0]][idx[1:]] for idx in key])
                    V.append((sum_delta, key, value))
                min_val, V_atom, gamma_max = min(V)

            # Exact line search
            numerator = denominator = 0.0
            smv_lookup = {}
            for idx in S_atom:
                numerator += (1.0/mu)*Y[idx[0]][idx[1:]] + W_1[idx[0]][idx[1:]] - W_2[idx[0]][idx[1:]]
                smv_lookup[idx] = 1.0
                denominator += mu
            for idx in V_atom:
                numerator -= (1.0/mu)*Y[idx[0]][idx[1:]] + W_1[idx[0]][idx[1:]] - W_2[idx[0]][idx[1:]]
                if idx not in smv_lookup:
                    denominator += mu
                else:
                    denominator -= mu
            gamma = gamma_max
            if not denominator < 10e-6:
                gamma = numerator/denominator
            if REINIT_W_ZERO_TOGGLE and fw_iter == 0:
                gamma = 1.0
            gamma = min(max(gamma, 0.0), gamma_max)
            print("gamma:", gamma, "gamma max:", gamma_max)
            
            # Update W_2
            for idx in S_atom:
                n,i,j,d,m = idx
                W_2[n][idx[1:]] += gamma
                if m == INSERTION:
                    mat_insertion[j][d] -= (max(0.0, delta[n][idx[1:]])-max(0.0, delta[n][idx[1:]]-mu*gamma))
                else:
                    tensor[j][d][util.move2T3idx(m)] -= (max(0.0, delta[n][idx[1:]])-max(0.0, delta[n][idx[1:]]-mu*gamma))
                delta[n][idx[1:]] -= mu * gamma
            for idx in V_atom:
                n,i,j,d,m = idx
                W_2[n][idx[1:]] -= gamma
                if m == INSERTION:
                    mat_insertion[j][d] -= (max(0.0, delta[n][idx[1:]])-max(0.0, delta[n][idx[1:]]+mu*gamma))
                else:
                    tensor[j][d][util.move2T3idx(m)] -= (max(0.0, delta[n][idx[1:]])-max(0.0, delta[n][idx[1:]]+mu*gamma))
                delta[n][idx[1:]] += mu * gamma

            if len(alpha_lookup) == 0:
                alpha_lookup[S_atom] = 1.0
            else:
                if S_atom in alpha_lookup:
                    alpha_lookup[S_atom] += gamma
                else:
                    alpha_lookup[S_atom] = gamma
                if alpha_lookup[V_atom] - gamma < 1e-6:
                    alpha_lookup.pop(V_atom)
                else:
                    alpha_lookup[V_atom] -= gamma
        
        recSeq = [t.acidB for t in trace]
        return recSeq


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
                            C[n][i][j][k][m] += PERB_EPS * (np.random.rand())
                            #print((n,i,j,k,m), allSeqs[n][i], C[n][i][j][k][m])
        return C
    
    


    def tensor5D_init(self, init_T2):
        tensor5D = []
        for i in range(self.numSeq):
            tmp_tensor = np.zeros((len(self.allSeqs[i]),init_T2, NUM_DNA_TYPE, NUM_MOVEMENT))
            tensor5D.append(tmp_tensor)
        return tensor5D