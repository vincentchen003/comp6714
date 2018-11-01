import torch
from config import config
from torch.nn import functional as F
from torch import Tensor
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
_config = config()

def check_dou(list, i):
    if (i < len(list) - 1):
        if (list[i + 1][0] == "I"):
            return True
        else:
            return False
    else:
        return False


def evaluate(golden_list, predict_list):
    length = len(golden_list)
    fn = 0
    fp = 0
    tp = 0
    for j in range(length):
        i = 0
        while (i < len(golden_list[j])):
            if (check_dou(golden_list[j], i) or check_dou(predict_list[j], i)):
                if (golden_list[j][i] == predict_list[j][i] and golden_list[j][i + 1] == predict_list[j][i + 1]):
                    tp += 1
                    i += 2
                else:
                    if (golden_list[j][i] == "O" and golden_list[j][i + 1] == "O"):
                        fp += 1
                        i += 2
                    elif (predict_list[j][i] == "O" and predict_list[j][i + 1] == "O"):
                        fn += 1
                        i += 2
                    else:
                        fp += 1
                        fn += 1
                        i += 2
            else:
                
                if (golden_list[j][i] != predict_list[j][i]):
                    if (golden_list[j][i] == "O"):
                        fp += 1
                        i += 1
                    elif (predict_list[j][i] == "O"):
                        fn += 1
                        i += 1
                    else:
                        fp += 1
                        fn += 1
                        i += 1
                else:
                    if (golden_list[j][i] == "O"):
                        i += 1
                    else:
                        tp += 1

                        i += 1
    if(tp!=0):                       
        P= tp/(tp+fp)
        R= tp/(tp+fn)
        F = 0       
        F = (2*P*R)/(P+R)
    else:
        F=0.0
    
    return float('%.3f'% F)


def new_LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hx,cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)
    cy = (forgetgate * cx) + ((1-forgetgate) * cellgate)
    hy = outgate * torch.tanh(cy)
    return hy, cy


def get_char_sequence(model, batch_char_index_matrices, batch_word_len_lists):
#reduce the dim in matrix and list
    dim1,dim2,dim3 = batch_char_index_matrices.size()
    batch_word_len_lists = batch_word_len_lists.reshape(dim1*dim2)
    new_char_matrix = batch_char_index_matrices.reshape(dim1*dim2,dim3)
####
    
    input_char_embeds = model.char_embeds(new_char_matrix)
    perm_idx, sorted_batch_word_len_lists = model.sort_input(batch_word_len_lists)
    sorted_input_embeds = input_char_embeds[perm_idx]
    _, desorted_indices = torch.sort(perm_idx, descending=False)

    output_sequence = pack_padded_sequence(sorted_input_embeds, lengths=sorted_batch_word_len_lists.data.tolist(), batch_first=True)
    output_sequence, state = model.char_lstm(output_sequence)
    output_sequence_t0 = state[0][0][desorted_indices]
    output_sequence_tc = state[0][1][desorted_indices]
    final_sequence = torch.cat([output_sequence_t0, output_sequence_tc], dim=-1)
    final_sequence = final_sequence.reshape(dim1,dim2,100)

    return final_sequence
