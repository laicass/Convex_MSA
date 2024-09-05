import numpy as np
from config import *
import util

def cube_smith_waterman_GPU(M, C, data_seq):
    # initialise
    print("gpu")
    np.set_printoptions(precision=2, suppress=True)
    data_seq_idx = np.array(list(map(util.dna2T3idx, data_seq)))
    trace = []
    T1, T2, T3 = C.shape[:3]
    score_tensor = np.zeros((T1, T2, T3))
    ans_idx_tensor = np.ones((T1, T2, T3), dtype=int) * -1
    action_tensor = np.ones((T1, T2, T3), dtype=int) * np.inf
    acidA_tensor = np.zeros((T1, T2, T3), dtype=int)
    acidB_tensor = np.zeros((T1, T2, T3), dtype=int)

    score_tensor[:,:,4] = np.inf
    score_tensor[0,0,:] = np.inf
    acc_ins_cost = C[0][0][4][11] + M[0][0][4][11]
    acc_ins_cost += np.cumsum(C[1:,0,4,0] + M[1:,0,4,0])

    score_tensor[1:,0,4] = acc_ins_cost
    ans_idx_tensor[1:,0,4] = INSERTION
    action_tensor[1:,0,4] = INSERTION
    acidA_tensor[1:,0,4] = data_seq_idx[1:]
    acidB_tensor[1:,0,4] = util.dna2T3idx(GAP_NOTATION)

    score_tensor[0,0,4] = C[0][0][4][11] + M[0][0][4][11]
    ans_idx_tensor[0,0,4] = MATCH_START
    action_tensor[0,0,4] = MATCH_START
    acidA_tensor[0,0,4] = util.dna2T3idx('*')
    acidB_tensor[0,0,4] = util.dna2T3idx('*')

    # start loop
    # 1a. get insertion score
    ins_score_tensor = np.full(score_tensor.shape, np.inf)
    ins_score_tensor[1:,:,:] = score_tensor[:-1,:,:]
    ins_score_tensor += M[:,:,:,INS_BASE_IDX] + C[:,:,:,INS_BASE_IDX]
    ins_score_tensor = ins_score_tensor[:,:,:,np.newaxis]

    # 1b. get deletion score
    del_score_tensor = np.full((T1, T2, T3, NUM_DNA_TYPE), np.inf)
    del_score_tensor[:,1:,:,:] = score_tensor[:,:-1,:].reshape(T1,T2-1,1,NUM_DNA_TYPE) + \
        M[:,1:,:,DEL_BASE_IDX:DEL_BASE_IDX+T3].transpose(0,1,3,2) + \
        C[:,1:,:,DEL_BASE_IDX:DEL_BASE_IDX+T3].transpose(0,1,3,2)
    
    # 1c. get max match/mismatch score
    mth_score_tensor = np.full((T1, T2, T3, NUM_DNA_TYPE), np.inf)
    mth_score_tensor[1:,1:,:,:] = score_tensor[:-1,:-1,:].reshape(T1-1,T2-1,1,NUM_DNA_TYPE) + \
        M[1:,1:,:,MTH_BASE_IDX:MTH_BASE_IDX+T3].transpose(0,1,3,2) + \
        C[1:,1:,:,MTH_BASE_IDX:MTH_BASE_IDX+T3].transpose(0,1,3,2)
    
    # 1d. get optimal action for the current cell
    score_stack = np.concatenate((ins_score_tensor, del_score_tensor, mth_score_tensor), axis=-1)

    min_ansid_stack = np.argmin(score_stack, axis=-1)
    min_score_stack = np.min(score_stack, axis=-1)
    min_ansid_stack[min_score_stack >= MAX_DOUBLE] = -1

    min_action_stack = np.ones_like(min_ansid_stack, dtype=int)*6
    min_action_stack[min_ansid_stack == INS_BASE_IDX] = INSERTION

    action_list = np.array(list(util.action2str.keys()))
    k_grid = np.broadcast_to(np.arange(T3), (T1, T2, T3))

    del_condition = (DEL_BASE_IDX <= min_ansid_stack) & (min_ansid_stack < MTH_BASE_IDX)
    min_action_stack = np.where(del_condition, action_list[DEL_BASE_IDX+k_grid], min_action_stack)

    mth_condition = (MTH_BASE_IDX <= min_ansid_stack) & (min_ansid_stack < NUM_MOVEMENT)
    min_action_stack = np.where(mth_condition, action_list[MTH_BASE_IDX+k_grid], min_action_stack)
    
    # 1e. assign the optimal score/action to the cell
    mask = np.ones((T1, T2, T3), dtype=bool)
    mask[0,0,:] = False
    mask[:,:,4] = False

    score_tensor[mask] = min_score_stack[mask]
    action_tensor[mask] = min_action_stack[mask]
    ans_idx_tensor[mask] = min_ansid_stack[mask]

    acidA_tensor[action_tensor == INSERTION] = np.broadcast_to(data_seq_idx[:,np.newaxis,np.newaxis], action_tensor.shape)[action_tensor == INSERTION]
    acidB_tensor[action_tensor == INSERTION] = util.dna2T3idx(GAP_NOTATION)

    del_condition = (DEL_BASE_IDX <= action_tensor) & (action_tensor < MTH_BASE_IDX)
    acidA_tensor[del_condition] = util.dna2T3idx(GAP_NOTATION)
    acidB_tensor[del_condition] = action_tensor[del_condition] - DEL_BASE_IDX

    mth_condition = (MTH_BASE_IDX <= action_tensor) & (action_tensor < NUM_MOVEMENT)
    acidA_tensor[mth_condition] = np.broadcast_to(data_seq_idx[:,np.newaxis,np.newaxis], action_tensor.shape)[mth_condition]
    acidB_tensor[mth_condition] = action_tensor[mth_condition] - MTH_BASE_IDX
    

    for i in range(T1):
        for j in range(T2):
            for k in range(T3):
                if (i == 0 and j == 0) or k == 4:
                    continue
                #print(action_tensor[i,j,k], acidA_tensor[i,j,k], acidB_tensor[i,j,k])
                print(f"min_ansid = {min_ansid_stack[i,j,k]}, min_score = {min_score_stack[i,j,k]}, min_action = {min_action_stack[i,j,k]}")
    
    exit()
