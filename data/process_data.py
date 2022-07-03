import os
import numpy as np
import math
import pandas as pd
try:
    from data import textgrid as tg
except:
    import textgrid as tg

SR = 16000

def get_min_max_frame_index(xmin, xmax, window_size):
    T_FRAME_SEC = window_size * 1000
    return math.floor(xmin / T_FRAME_SEC), math.floor(xmax / T_FRAME_SEC)

def interval_in_word_tier(interval, word_tier, ignore_multiple=False):
    '''
    returns given interval (from another tier) in word_tier. If the equivalent time interval has no mark, returns None.
    '''
    segment_in_word_tier = None
    int_interval = pd.Interval(interval.minTime, interval.maxTime)
    for wt in word_tier.intervals:
        # check if intersect
        wt_interval =  pd.Interval(wt.minTime, wt.maxTime)
        if wt_interval.overlaps(int_interval)  and wt.mark != '':
            if not ignore_multiple:
                assert segment_in_word_tier is None, 'multiple intersecting intervals'
                segment_in_word_tier = wt
            else:
                if  segment_in_word_tier is None:
                    segment_in_word_tier = wt
                else:
                    new_interval = tg.Interval(segment_in_word_tier.minTime, wt.maxTime, f"{segment_in_word_tier.mark},{wt.mark}")
                    segment_in_word_tier = new_interval
            
    return segment_in_word_tier

def is_creak_label(interval, word_tier, ignore_multiple=False):
    # only consider creaks marked within a word in the word_tier
    return interval.mark == 'c' and interval_in_word_tier(interval, word_tier, ignore_multiple) is not None

def is_sil_label(interval, word_tier, ignore_multiple=False):
    # ignore all segments not within the word_tier.
    segment_in_word_tier = interval_in_word_tier(interval, word_tier, ignore_multiple)
    if segment_in_word_tier and interval.maxTime > segment_in_word_tier.maxTime:
        return True
    if segment_in_word_tier and interval.minTime < segment_in_word_tier.minTime and interval.mark != segment_in_word_tier.mark:
        return True
    return segment_in_word_tier is None

def label_frame(label, frame_vec, xmin, xmax, window_size, label_neg = False):
    '''
    :param label: label of given time frame
    :param frame_vec: frames label vector
    :param xmin: frame's min time - in sec
    :param xmax: frame's max time - in sec
    :return:
    '''
    min_frame_index, max_frame_index = get_min_max_frame_index(xmin, xmax, window_size)
    T_FRAME_SEC = window_size * 1000
    # check indexes on both edges, since they intersect with other time frames
    if T_FRAME_SEC - xmin % T_FRAME_SEC > T_FRAME_SEC/2 and not (label_neg and frame_vec[min_frame_index] == -2):
        frame_vec[min_frame_index] = label
    if xmax % T_FRAME_SEC > T_FRAME_SEC/2 and not (label_neg and frame_vec[max_frame_index] == -2):
        frame_vec[max_frame_index] = label
    # set indexes in the range as creaky
    if label_neg:
        for i in range(len(frame_vec[min_frame_index + 1:max_frame_index])):
            if frame_vec[min_frame_index + 1+i] != -2:
                frame_vec[min_frame_index + 1+i] = label
    else:
        frame_vec[min_frame_index + 1:max_frame_index] = label

def iterate_label_tier_intervals_all(labels_tier, word_tier, phn_tier, labels_count, phn_dict,window_size):
    frame_2_creaky_label = np.zeros(labels_count)
    frame_2_voice_label = np.zeros(labels_count)
    for interval in labels_tier.intervals:
        min_time, max_time = interval.minTime*1000,interval.maxTime*1000
        if interval.mark == 'c':
            label_frame(1, frame_2_creaky_label, min_time, max_time, window_size)
    
    for interval in phn_tier.intervals:
        min_time, max_time = interval.minTime*1000,interval.maxTime*1000
        if is_voice_label(interval, word_tier, phn_dict, ignore_multiple=True):
            label_frame(1, frame_2_voice_label, min_time, max_time, window_size)

    for interval in labels_tier.intervals:
        min_time, max_time = interval.minTime*1000,interval.maxTime*1000
        if interval.mark == '':
            label_frame(-1, frame_2_creaky_label, min_time, max_time, window_size)
            label_frame(-1, frame_2_voice_label, min_time, max_time, window_size)

    for interval in word_tier.intervals:
        min_time, max_time = interval.minTime*1000,interval.maxTime*1000
        if interval.mark == 'sp' or is_sil_label(interval, word_tier):
            label_frame(-1, frame_2_creaky_label, min_time, max_time, window_size)

    
    not_voice_creaky = np.logical_and(frame_2_creaky_label ==1, frame_2_voice_label == 0)
    not_voice_creaky_idx = np.where(not_voice_creaky)
    frame_2_creaky_label[not_voice_creaky_idx] = 0
    return frame_2_creaky_label, frame_2_voice_label

def is_voice_label(interval, word_tier, phn_dict, ignore_multiple=False):
    # only consider creaks marked within a word in the word_tier
    mark = interval.mark.lower()
    if mark and mark[-1].isdigit():
        mark = mark[:-1]
    if mark not in phn_dict:
        return False
    return (phn_dict[mark] == 'v'  or  phn_dict[mark] == 'g' or  phn_dict[mark] == 'n') and interval_in_word_tier(interval, word_tier, ignore_multiple) is not None

def get_file_labels_allstar(spkr_tg_filename, phn_dict, window_size, creak_tier_name="creak-gold", len=None ):
    tg_file = os.path.basename(spkr_tg_filename)
    spkr_tg = tg.TextGrid.fromFile(spkr_tg_filename)
    file_et = spkr_tg.maxTime
    labels_count = math.ceil(file_et / window_size)
    if len is not None:
        assert len == labels_count or len == labels_count-1
        if labels_count > len:
            labels_count -= 1
    # tagging_lvl is 'ipp'
    labels_tier = spkr_tg.getFirst(creak_tier_name)
    word_tier = spkr_tg.getFirst('Speaker - word')

    
    phn_tier = spkr_tg.getFirst('Speaker - phone')
    frame_2_label = iterate_label_tier_intervals_all(labels_tier, word_tier, phn_tier, labels_count, phn_dict, window_size)
    return frame_2_label
    

def find_creaky_section_area(interval, tier, thr = 0):
    for wt in tier.intervals:
        # check if intersect
        wt_interval =  pd.Interval(wt.minTime, wt.maxTime)
        if wt_interval.overlaps(interval)  and wt.mark == 'c':
            intersection = min(interval.right,wt_interval.right) - max(interval.left,wt_interval.left)
            if intersection / interval.length >= thr: # default- if the intervals have some overlaping area, count as exist
                return True
            
    return False

def calculate_file_acc(tier,word_tier,pred_creaky,target_creaky, ignore_multiple, thr=0):
    tier_count = 0
    creak_tier_count = 0
    corect_tier_c = 0
    fp_tier_c = 0
    fn_tier_nc = 0
    for interval in tier.intervals:
        if interval.mark != '' and interval_in_word_tier(interval, word_tier, ignore_multiple) is not None:
            tier_count +=1
            int_interval = pd.Interval(interval.minTime, interval.maxTime)
            pred_flag = find_creaky_section_area(int_interval,pred_creaky, thr=0)
            target_flag = find_creaky_section_area(int_interval,target_creaky, thr=0)
            if target_flag:
                creak_tier_count +=1
            if target_flag and pred_flag:
                corect_tier_c +=1
            elif pred_flag and not target_flag:
                fp_tier_c +=1
            elif not pred_flag and target_flag:
                fn_tier_nc +=1

    
    return tier_count,creak_tier_count,corect_tier_c,fp_tier_c, fn_tier_nc

def calculate_file_acc_allstar(tier,word_tier,pred_creaky,target_creaky, ignore_multiple):
    tier_count = 0
    creak_tier_count = 0
    corect_tier_c = 0
    fp_tier_c = 0
    fn_tier_nc = 0
    for interval in tier.intervals:
        if interval.mark != '' and interval_in_word_tier(interval, word_tier, ignore_multiple) is not None:
            tier_count +=1
            int_interval = pd.Interval(interval.minTime, interval.maxTime)
            pred_flag = find_creaky_section_area(int_interval,pred_creaky, thr=0)
            target_flag = find_creaky_section_area(int_interval,target_creaky, thr=0)
            if target_flag:
                creak_tier_count +=1
            if target_flag and pred_flag:
                corect_tier_c +=1
            elif pred_flag and not target_flag:
                fp_tier_c +=1
            elif not pred_flag and target_flag:
                fn_tier_nc +=1

    
    return tier_count,creak_tier_count,corect_tier_c,fp_tier_c, fn_tier_nc
