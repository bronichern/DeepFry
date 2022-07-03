import torch.nn as nn
import torch
import numpy as np
from data import textgrid, process_data
import os

NOT_CREAKY=0
CREAKY=1

phn_dict = {"iy":"v","ih":"v" ,"eh":"v","ey" :"v" , "ae" :"v" ,"aa" :"v", "aw" :"v", "ay" :"v","ah" :"v", "ao" :"v" ,    
            "oy" :"v","ow" :"v","uh" :"v" , "uw" :"v","ux" :"v" ,"er" :"v","ax" :"v","ix" :"v"  ,"axr" :"v","ax-h" :"v", 
            "m" :"n","n"  :"n", "ng" :"n" , "em" :"n" ,"en" :"n","eng":"n","nx":"n" ,
             "s" : "f" , "sh": "f" ,"z" : "f"  ,"zh": "f", "f" : "f", "th": "f","v": "f" , "dh" :"f"  ,  
             "l": "g","r": "g" ,"w" : "g" , "y" : "g","hh" : "g" ,"hv" : "g" ,"el": "g"  ,
             "jh" : "a", "ch" :"a" ,
             "b": "s", "d": "s" ,"g" : "s","p" : "s" ,"t": "s" , "k" : "s", "dx" : "s" , "q" : "s" , 
             "sp":"sp"}    

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad
    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """

    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size).to(vec.device)], dim=dim)

def padd_list_tensors(targets, targets_lens, dim):

    target_max_len = max(targets_lens) if type(targets_lens) == list else targets_lens
    padded_tensors_list = []
    for tensor in targets:
        pad = pad_tensor(tensor, target_max_len, dim)
        padded_tensors_list.append(pad)
    padded_tensors = torch.stack(padded_tensors_list)
    return padded_tensors

def get_section(preds_array, start):
    change_value = np.diff(preds_array) 
    change_value_idx =  np.argwhere(change_value != 0)
    sections_list = []
    start_idx = 0
    for idx in change_value_idx:
        idx = idx[0]
        mark = preds_array[idx]
        item_len = idx - start_idx +1

        sections_list.append([start_idx + start, idx+1 + start,mark, item_len])
        start_idx = idx+1
    if start_idx != len(preds_array):

        mark = preds_array[start_idx]
        item_len = len(preds_array) - start_idx 
        sections_list.append([start_idx + start, len(preds_array)+ start,mark, item_len])
    return sections_list

def add_tier(tier_name, sections, description_type,f_window, max_time=-1, empty_desc="m"):
    tier = textgrid.IntervalTier(name=tier_name, minTime=0)
    prev_end = sections[0][0]* f_window
    for item in sections:
        start_item, end_item, mark, item_len = item
        start_sec = (start_item ) * f_window
        end_sec = (end_item ) * f_window
        if mark == 1:
            description = f"{description_type}"
        else:
            description = "empty_desc" 
        if prev_end != start_sec:
            tier.add(prev_end, start_sec, "")

        tier.add(start_sec, end_sec, description)
        prev_end = end_sec
    if max_time!= -1:
        if prev_end != max_time:
            if prev_end > max_time and prev_end-max_time <= f_window:
                max_time = prev_end
            tier.add(prev_end, max_time, "")
    return tier

def process_sections(sections):

    new_sections = [sections[0]]
    for section in sections[1:]:
        prev_start, prev_end, prev_mark, prev_len = new_sections[-1]
        current_start, current_end, current_mark, current_len = section
        if prev_mark != current_mark:
            if prev_len == 1:
                if prev_end == current_start:
                    if len(new_sections) > 2:
                        new_sections.pop()
                        prev_start, prev_end, prev_mark, prev_len = new_sections.pop()
                        new_sections.append([prev_start, current_end, current_mark, current_len + prev_len+1])
                        continue
                new_sections.append(section)
                continue
            if current_start - prev_end == 1:
                new_sections.append([prev_end, current_end, current_mark, current_len +1])
                continue
            if prev_mark != current_mark:
                new_sections.append(section)
                continue
        if current_mark == 1 and current_start - prev_end < 2: 
            new_sections.pop()
            new_sections.append([prev_start, current_end, current_mark, current_len + prev_len])
        else:
            new_sections.append(section)

    return new_sections
        

def create_textgrid(predict_dict, output,f_window, custom):
  
    if not os.path.exists(output):
        os.makedirs(output)
    for filename, predictions in predict_dict.items():
        
        prev_textgrid = textgrid.TextGrid.fromFile(filename)
        
        pred_creaky,target_creaky = create_tires(predictions,f_window, max_time=prev_textgrid.maxTime)
        

        basename = os.path.basename(filename)
        new_filename = os.path.join(output , basename)
        
        prev_textgrid.append(pred_creaky)
        if not custom:
            prev_textgrid.append(target_creaky)
        prev_textgrid.write(new_filename)

def create_tires(predictions,f_window,max_time=-1):
    pred_creaky_sections = []
    label_creaky_sections = []

    for start, pred_creaky, label_creaky in predictions:

        pred_creaky_sections.extend(get_section(pred_creaky, start))
        label_creaky_sections.extend(get_section(label_creaky, start))

    pred_creaky_sections = process_sections(pred_creaky_sections)
    label_creaky_sections = process_sections(label_creaky_sections)
    pred_creaky = add_tier("pred-creaky",pred_creaky_sections, "c",f_window, max_time)
    target_creaky = add_tier("target-creaky",label_creaky_sections, "c",f_window, max_time)

    return pred_creaky,target_creaky

def word_pho_acc(predict_dict,f_window):
    words_count = []
    creak_words_count= []
    corect_words_c = []
    fp_words_c = []
    fn_words_c = []
    phons_count = []
    creak_phons_count = []
    corect_phons_c = []
    fp_phons_c = []
    fn_phons_c = []
    
    for filename, predictions in predict_dict.items():
        pred_creaky, target_creaky = create_tires(predictions,f_window)
        basename = os.path.basename(filename)
        file_number = basename[7:basename.rfind(".")]
        spkr_tg = textgrid.TextGrid.fromFile(filename)

        s10word_tier = spkr_tg.getFirst('Speaker - word')
        s10phn_tier = spkr_tg.getFirst('Speaker - phone')
        word_tier = s10word_tier
        ignore_multiple = True
                                                                                                                                        
        file_words_count,creak_file_words_count,corect_file_words_c,fp_file_words_c, fn_file_words_c = process_data.calculate_file_acc(s10word_tier,word_tier,pred_creaky,target_creaky, ignore_multiple)
        words_count.append(file_words_count)
        creak_words_count.append(creak_file_words_count)
        corect_words_c.append(corect_file_words_c)
        fp_words_c.append(fp_file_words_c)
        fn_words_c.append(fn_file_words_c)

        file_phons_count,creak_file_phons_count,corect_file_phons_c,fp_file_phons_c, fn_file_phons_c = process_data.calculate_file_acc(s10phn_tier,word_tier,pred_creaky,target_creaky,ignore_multiple, 0)
        phons_count.append(file_phons_count)
        creak_phons_count.append(creak_file_phons_count)
        corect_phons_c.append(corect_file_phons_c)
        fp_phons_c.append(fp_file_phons_c)
        fn_phons_c.append(fn_file_phons_c)

    phone_precision = sum(corect_phons_c) / (sum(corect_phons_c) + sum(fp_phons_c)) if (sum(corect_phons_c) + sum(fp_phons_c))> 0 else 0 
    phone_recall = sum(corect_phons_c) / (sum(corect_phons_c) + sum(fn_phons_c)) if (sum(corect_phons_c) + sum(fn_phons_c))  > 0 else 0 

    phone_precision_by_file = np.mean(np.array(corect_phons_c) / (np.array(corect_phons_c) + np.array(fp_phons_c)))
    phone_recall_by_file = np.mean(np.array(corect_phons_c) / (np.array(corect_phons_c) + np.array(fn_phons_c)))

    word_precision = sum(corect_words_c) / (sum(corect_words_c) + sum(fp_words_c)) if (sum(corect_words_c) + sum(fp_words_c))>0 else 0
    word_recall = sum(corect_words_c) / (sum(corect_words_c) + sum(fn_words_c)) if (sum(corect_words_c) + sum(fn_words_c)) > 0 else 0


    word_precision_by_file = np.mean(np.array(corect_words_c) / (np.array(corect_words_c) + np.array(fp_words_c)))
    word_recall_by_file = np.mean(np.array(corect_words_c) / (np.array(corect_words_c) + np.array(fn_words_c)))

    return word_precision, word_recall, phone_precision, phone_recall

def find_pairs(pred_sections, target_sections):
    pred_sections = np.array(pred_sections)
    target_sections = np.array(target_sections)
    pairs = []
    weird = []
    for idx, (pred_start, pred_end, ptype, pred_len) in enumerate(pred_sections):
        min_start_idx = np.abs(target_sections[:, 0]  - pred_start).argmin()
        min_end_idx = np.abs(target_sections[:, 1]  - pred_end).argmin()
        if min_start_idx == min_end_idx:
            pairs.append([idx, min_start_idx])
        else:
            weird.append(pred_sections[idx])
            one_target = target_sections[min_start_idx]
            two_target = target_sections[min_end_idx]
            un_one = min(one_target[1], pred_end) - max(one_target[0], pred_start)
            un_two = min(two_target[1], pred_end) - max(two_target[0], pred_start)
            if un_one> un_two and un_one > 0:
                pairs.append([idx, min_start_idx])
            elif un_two> un_one and un_two > 0:
                pairs.append([idx, min_end_idx])

    target_pairs = [x[1] for x in pairs]
    double_target_use = []
    idx = 0
    while idx < len(pairs): # search for target that fit to more than one prediction
        pred_idx, target_idx = pairs[idx]
        current_double_use = [idx]
        j = idx+1
        for next_idx in range(j, len(target_pairs)):
            if target_idx == target_pairs[next_idx]:
                current_double_use.append(next_idx)
                idx+=1
        if len(current_double_use) > 1:
            double_target_use.append([target_idx,current_double_use])
        idx+=1
    remove_list = []
    for target_idx, idx_list in double_target_use: # choose the pair with thw largest intersection as the currect pair
        best_uni = - np.inf
        best_uni_idx = -1
        start_t = target_sections[target_idx][0]
        end_t = target_sections[target_idx][1]
        for idx in idx_list:
            pred_idx = pairs[idx][0]
            # pred_idx = idx
            start_p = pred_sections[pred_idx][0]
            end_p = pred_sections[pred_idx][1]
            uni = min(end_t, end_p) - max(start_t, start_p)
            if uni > best_uni:
                best_uni = uni
                best_uni_idx = idx
        idx_list.remove(best_uni_idx)
        remove_list.extend(idx_list)

    new_pairs = []
    for idx in range(len(pairs)):
        if idx not in remove_list:
            new_pairs.append(pairs[idx])

    return new_pairs

def iou(pred_item, target_item): 

    if target_item[0] > pred_item[1] or target_item[1] < pred_item[0]: # ---- ----
        return 0  
    overlap_start = max(pred_item[0],target_item[0])
    union_start = min(pred_item[0],target_item[0])
    overlap_end = min(pred_item[1],target_item[1])
    union_end = max(pred_item[1],target_item[1])

    diff = overlap_end - overlap_start
    overlap = diff if diff > 0 else 0
    union  = union_end - union_start
    iou_val = overlap / union
    return iou_val

def actual_accuracy_tolerance(preds, targets, mark, len_tolerance, union_tolerance):

    preds_sections = process_sections(get_section(preds, 0))
    target_sections = process_sections(get_section(targets, 0))

    pred = [x for x in  preds_sections if x[2]==mark]
    target = [x for x in  target_sections if x[2]==mark]
    pairs = find_pairs(pred, target)

    len_tolerance_count = 0
    union_tolerance_count = 0
    intersection = 0
    overlap_array = []
    iou_array = []
    pred_len_array  = []
    target_len_array  = []
    for pred_idx, target_idx in pairs:
        pred_item = pred[pred_idx]
        target_item = target[target_idx]
        iou_val = iou(pred_item, target_item)
        iou_array.append(iou_val)
        if min(pred_item[1], target_item[1]) - max(pred_item[0], target_item[0]) < 0: # check that some intersection exist
            continue
        intersection += 1
        pred_len_array.append(pred_item[3])
        target_len_array.append(target_item[3])
        if abs(pred_item[3] - target_item[3]) <= len_tolerance:
            len_tolerance_count += 1

        overlap = min(target_item[1], pred_item[1]) - max(target_item[0], pred_item[0])
        overlap_array.append(overlap/ target_item[3])
        if overlap/ target_item[3] >= union_tolerance:
            union_tolerance_count += 1
    # print("************* tolerance:{} ******************".format(tolerance))
    print("len_tolerance:{}, union_tolerance:{}".format(len_tolerance, union_tolerance))
    print('mean: pred len:{}, target len :{}, overlap:{}, iou:{} \n'.format(np.mean(pred_len_array), np.mean(target_len_array), np.mean(overlap_array), np.mean(iou_array)))
    print('len recall:{}, union recall:{}, intersection recall:{} \n'.format(len_tolerance_count/len(target), union_tolerance_count/len(target), intersection/len(target)))
