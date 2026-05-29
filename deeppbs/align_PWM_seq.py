import numpy as np
from deeppbs.nn.metrics import IC_weighted_PCC

def _validateContactMask(mask, length, name):
    if mask is None:
        return None
    mask = np.asarray(mask).astype(bool)
    if mask.shape[0] != length:
        raise ValueError("{} length must match aligned sequence length".format(name))
    return mask

def ungappedAlign(ml, ms, gt, ml_contact_mask=None, ms_contact_mask=None): ## ungapped alignment maximising dot product, needs Length X 4 arrays
        
    ml_contact_mask = _validateContactMask(ml_contact_mask, ml.shape[0], "ml_contact_mask")
    ms_contact_mask = _validateContactMask(ms_contact_mask, ms.shape[0], "ms_contact_mask")
    use_contact_constraint = ml_contact_mask is not None or ms_contact_mask is not None

    max_score = -9999
    max_contact_count = -1
    opt_i = 0
    opt_j = 0
    opt_k = 0
    found = False
    l = ml.shape[0]
    s = ms.shape[0]
    for i in range(0,s):
        for k in range(1, s-i+1):   ### k overlap length
            
            for j in range(0,l - k+1):
                #score = np.sum(ms[i:i+k,:]* ml[j:j+k,:]) # dot product scoring
                contact_count = 0
                if ml_contact_mask is not None:
                    contact_count += ml_contact_mask[j:j+k].sum()
                if ms_contact_mask is not None:
                    contact_count += ms_contact_mask[i:i+k].sum()
                if use_contact_constraint and contact_count == 0:
                    continue
                
                score = 0

                if np.array_equal(ml, gt): #IC_weighted_PCC scoring
                    idx = j
                else:
                    idx = i
                for col in range(k):
                    col_score, _ = IC_weighted_PCC(ms[i:i+k,:][col,:], ml[j:j+k,:][col,:], gt=gt[idx:idx+k,:][col,:])
                    score += col_score

                
                
                if(score > max_score or (np.isclose(score, max_score) and contact_count > max_contact_count)):
                    max_score = score
                    max_contact_count = contact_count
                    opt_i = i
                    opt_j = j
                    opt_k = k
                    found = True
    if use_contact_constraint and not found:
        raise ValueError("No alignment window overlaps the contact mask")
    return opt_i, opt_j, opt_k, max_score

def alignPWMSeq(pwm, seq, contact_mask=None):
    if(pwm.shape[0] > seq.shape[0]):
        seq_start, pwm_start, k, max_score = ungappedAlign(pwm, seq, pwm, ms_contact_mask=contact_mask)
    else:
        pwm_start, seq_start, k, max_score = ungappedAlign(seq, pwm, pwm, ml_contact_mask=contact_mask)
        
    return pwm_start, seq_start, k, max_score
