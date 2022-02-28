import os, sys
from os.path import *
import csv, json, copy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt
import random


# get scores of ssfix, s3, capgen, opad

def add_patch_property(bugs_dict, bug_id, patch_name, property_name, property_value):
    if not bug_id in bugs_dict.keys(): bugs_dict[bug_id] = dict()
    patches_dict = bugs_dict[bug_id]
    if not patch_name in patches_dict.keys(): patches_dict[patch_name] = dict()
    patch_dict = patches_dict[patch_name]
    patch_dict[property_name] = property_value

def rank_correct_patch(patches_dict, tool):
    # the patches should contain at least one correct patch and one overfitting patch
    
    reverse = tool == 's3'
    correct_patches_dict = {k: v for k, v in patches_dict.items() if v['label'] == 'correct'}
    overfitting_patches_dict = {k: v for k, v in patches_dict.items() if v['label'] == 'overfitting'}
    if len(correct_patches_dict) > 0 and len(overfitting_patches_dict) == 0: return None
    if len(correct_patches_dict) ==0 and len(overfitting_patches_dict) > 0: return None
    correct_patch_score = max([float(patch_dict[tool]) for patch_dict in correct_patches_dict.values()])
    if reverse: correct_patch_score = min([float(patch_dict[tool]) for patch_dict in correct_patches_dict.values()])
    rank = 1
    for patch_name in patches_dict.keys():
        score = float(patches_dict[patch_name][tool])
        if not reverse:
            if score > correct_patch_score: rank += 1
        else:
            if score < correct_patch_score: rank += 1
    return rank
    
def rank_patches_per_bug(bugs_dict, tool):
    rank_dict = dict()
    for bug_id in bugs_dict:
        patches_dict = bugs_dict[bug_id]
        rank = rank_correct_patch(patches_dict, tool)
        if rank == None: continue
        # print(bug_id + ': ' + str(rank) + ' out of ' + str(len(patches_dict)))
        rank_dict[bug_id] = (rank, len(patches_dict), tool)
        
    return rank_dict
    
def get_top_N(score_label_list, correct_num, tool):
    top_N = list()
    others = list()
    tied = list()
    if tool == 'capgen' or tool == 'ssfix': reverse = True
    if tool == 's3': reverse = False
    sorted_score_label_list = sorted(score_label_list, key=lambda x: x[0], reverse=reverse)
    threshold = sorted_score_label_list[correct_num - 1][0]
    for pair in score_label_list:
        if pair[0] > threshold: 
            if reverse: top_N.append(pair)
            else: others.append(pair)
        if pair[0] < threshold: 
            if reverse: others.append(pair)
            else: top_N.append(pair)
        if pair[0] == threshold: tied.append(pair)
    
    if len(top_N) < correct_num and len(top_N) + len(tied) > correct_num:
        sampled = random.sample(sorted(tied), correct_num - len(top_N))
        top_N = top_N + sampled
        for data in sampled:
            tied.remove(data)
        others = others + tied
    elif len(top_N) + len(tied) == correct_num:
        top_N = top_N + tied
    assert len(top_N) == correct_num
    return top_N, others

def print_confusion_matrix_from_patches(patches, tool):
    score_label_list = list()
    correct_num = 0
    for bug_id in patches:
        patch_dict = patches[bug_id]
        for patch_name in patch_dict:
            score = float(patch_dict[patch_name][tool])
            label = patch_dict[patch_name]['label']
            score_label_list.append((score, label))
            if label == 'correct': correct_num += 1       
    
    print_confusion_matrix(score_label_list, correct_num, tool)     
            
def print_confusion_matrix(score_label_list, correct_num, tool): 
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    top_N, others = get_top_N(score_label_list, correct_num, tool)
    for element in top_N:
        if element[1] == 'correct':
            TN += 1
        else:
            FN += 1
    for element in others:
        if element[1] == 'correct':
            FP += 1
        else:
            TP += 1
    print("TN: " + str(TN))
    print("FN: " + str(FN))
    print("TP: " + str(TP))
    print("FP: " + str(FP))

    print("precision: " + str(TP / (TP + FP)))
    print("recall: " + str(TP / (TP + FN)))
    print("F1: " + str(TP / (TP + 1/2 * (FP + FN))))
    
def rank_dev_patches(dev_patches, tool_patches, tool):
    reverse = tool == 's3'
    rank_dict = dict()
    for bug_id in tool_patches:
        dev_patch_dict = list(dev_patches[bug_id].values())[0]
        dev_score = float(dev_patch_dict[tool])
        overfitting_patches_dict = {k: v for k, v in tool_patches[bug_id].items() if v['label'] == 'overfitting'}
        rank = 1
        if len(overfitting_patches_dict) == 0: continue
        for _, patch_dict in overfitting_patches_dict.items():
            overfit_patch_score = float(patch_dict[tool])
            if not reverse:
                if overfit_patch_score > dev_score: rank += 1
            else:
                if overfit_patch_score < dev_score: rank += 1
        rank_dict[bug_id] = (rank, len(overfitting_patches_dict) + 1, tool)
        # print('%s: developer patches ranking: %d out of %d' % (bug_id, ranking, len(overfitting_patches_dict) + 1))
    return rank_dict

@DeprecationWarning
def parse_ase_result(patch_list_path):
    # the patch list includes TPs, i.e., overfitting patches identified as overfitting patches
    patch_set = set()
    with open(patch_list_path) as f:
        patches = f.read().splitlines()
    for patch_name in patches:
        if patch_name.endswith('-plausible'): patch_name = patch_name.replace('-plausible', '')
        if patch_name.startswith('patch'):
            # this patch is from the five tools
            patch_id = patch_name.split('-')[0][5:]
            project, id, tool = patch_name.split('-')[1:4]
        else: 
            patch_id = '0'
            tool, project, id = patch_name.split('-')
        
        dlabel = 'Doverfitting'
        if patch_id == '0': subdir = 'Patches_ICSE'
        else: subdir = 'Patches_others'
        patch_dir = join(ASE_patch_dir, subdir, dlabel, tool, project, id)
        if patch_id != '0': patch_dir = join(patch_dir, patch_id)
        if not isdir(patch_dir): 
            patch_dir = patch_dir.replace(dlabel, 'Dcorrect')
            dlabel = 'Dcorrect'
        assert isdir(patch_dir)
        patch_set.append((project, id, tool, patch_id, dlabel))
    return patch_set

@DeprecationWarning
def store_ASE_opad_result(ase_patches):
    all_overfitting_patch_set = parse_ase_result('/home/junyang/PCC_repo/patch_correctness/ASE_Patches/overfitting_patches.txt')
    evo_opad_TP_patch_set = parse_ase_result('/home/junyang/PCC_repo/patch_correctness/ASE_Patches/evosuite_opad_true_overfitting.txt')
    rand_opad_TP_patch_set = parse_ase_result('/home/junyang/PCC_repo/patch_correctness/ASE_Patches/randoop_opad_true_overfitting.txt')
    evo_opad_overfitting_patch_set = parse_ase_result('/home/junyang/PCC_repo/patch_correctness/ASE_Patches/evosuite_opad_overfitting.txt')
    rand_opad_overfitting_patch_set = parse_ase_result('/home/junyang/PCC_repo/patch_correctness/ASE_Patches/randoop_opad_overfitting.txt')
    evo_opad_FP_patch_set = evo_opad_overfitting_patch_set - evo_opad_TP_patch_set
    rand_opad_FP_patch_set = rand_opad_overfitting_patch_set - rand_opad_TP_patch_set
    for patch in evo_opad_FP_patch_set:
        patch_name = '-'.join([patch[2], patch[3], patch[4]])
        add_patch_property(ase_patches, patch[0] + '-' + patch[1], patch_name, 'evo_opad_label', 'FP')
    
    for patch in rand_opad_FP_patch_set:
        patch_name = '-'.join([patch[2], patch[3], patch[4]])
        add_patch_property(ase_patches, patch[0] + '-' + patch[1], patch_name, 'rand_opad_label', 'FP')
        
    for patch in evo_opad_TP_patch_set:
        patch_name = '-'.join([patch[2], patch[3], patch[4]])
        add_patch_property(ase_patches, patch[0] + '-' + patch[1], patch_name, 'evo_opad_label', 'TP')
    
    for patch in rand_opad_TP_patch_set:
        patch_name = '-'.join([patch[2], patch[3], patch[4]])
        add_patch_property(ase_patches, patch[0] + '-' + patch[1], patch_name, 'rand_opad_label', 'TP')
    
def store_ASE_patches(ase_patches):
    with open(join(s3_capgen_dir, 'ASE_patch_score_capgen_s3.txt')) as f:
        lines = f.readlines()
        for line in lines[1:]:
            bug, s3, capgen = line.strip().split('\t')
            bug = bug.split('Dataset_Overfitting2.D')[1]
            if len(bug.split('.')) == 5: label, tool, project, id, patch_id = bug.split('.')
            else:
                assert len(bug.split('.')) == 4
                label, tool, project, id = bug.split('.')
                patch_id = '0'
            if patch_id == '0': sub_dir = 'Patches_ICSE'
            else: sub_dir = 'Patches_others'
            patch_dir = join(ASE_patch_dir, sub_dir, 'D' + label, tool, project, id)
            if patch_id != '0': patch_dir = join(patch_dir, patch_id)
            assert isdir(patch_dir), patch_dir
            bug_id = '-'.join([project, id])
            patch_name = '-'.join([tool, patch_id, 'D' + label])
            if isfile(join(patch_dir, 'NOT_PLAUSIBLE')): continue
            if isfile(join(patch_dir, 'MISLABEL')):
                if label == 'correct': label = 'overfitting'
                elif label == 'overfitting': label = 'correct'
            add_patch_property(ase_patches, bug_id, patch_name, 'capgen', capgen)
            add_patch_property(ase_patches, bug_id, patch_name, 's3', s3)
            add_patch_property(ase_patches, bug_id, patch_name, 'label', label)
            add_patch_property(ase_patches, bug_id, patch_name, 'overlapping', not isfile(join(patch_dir, 'NOT_OVERLAP')))
    
    with open(join(ssfix_dir, 'result_ASE_patches_full.csv'), newline='') as f:
        # lines = f.readlines()
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            tool, project, id, patch_id, line_in_buggy, line_in_patched, structural_score, conceptual_score, ssfix, correct = row
            if tool == 'tool': continue
            if correct == 'True': label = 'correct'
            else: label = 'overfitting'
            if patch_id == '0': sub_dir = 'Patches_ICSE'
            else: sub_dir = 'Patches_others'
            patch_dir = join(ASE_patch_dir, sub_dir, 'D' + label, tool, project, id)
            if patch_id != '0': patch_dir = join(patch_dir, patch_id)
            assert isdir(patch_dir), patch_dir
            bug_id = '-'.join([project, id])
            patch_name = '-'.join([tool, patch_id, 'D' + label])
            if isfile(join(patch_dir, 'NOT_PLAUSIBLE')): continue
            if isfile(join(patch_dir, 'MISLABEL')):
                if label == 'correct': 
                    label = 'overfitting'
                elif label == 'overfitting': label = 'correct'
            add_patch_property(ase_patches, bug_id, patch_name, 'ssfix', ssfix)
            assert ase_patches[bug_id][patch_name]['label'] == label, ase_patches[bug_id]
    # store_ASE_opad_result(ase_patches)

def merge_prapr_ase_patches(prapr_patches, ase_patches):
    merged_patches = copy.deepcopy(prapr_patches)
    ase_patches_filtered = dict()
    for bug_id in ase_patches:
        patches_dict = ase_patches[bug_id]
        for patch_name in patches_dict:
            patch_dict = patches_dict[patch_name]
            # Overlapping patches will be excluded when merging
            if patch_dict['overlapping']: continue
            for k, v in patch_dict.items():
                add_patch_property(merged_patches, bug_id, patch_name, k, v)
                add_patch_property(ase_patches_filtered, bug_id, patch_name, k, v)
                
    return merged_patches
    
def store_prapr_or_dev_patches(patches, csv_file, patch_root_dir):
    with open(join(s3_capgen_dir, csv_file), newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
        # for line in lines[1:]:
            project, id, mutant_id, toolASTDifferencing, toolCosine, toolStringDistance, toolVariable, toolSyntax, \
                toolSemantic, s3, capgen, correct = row
            if project == 'project': continue
            if correct == 'TRUE': label = 'correct'
            else: label = 'overfitting'
            patch_dir = join(patch_root_dir, project, id, mutant_id)
            assert isdir(patch_dir), patch_dir
            bug_id = '-'.join([project, id])
            if isfile(join(patch_dir, 'CANT_FIX')): continue
            add_patch_property(patches, bug_id, mutant_id, 's3', s3)
            add_patch_property(patches, bug_id, mutant_id, 'capgen', capgen)
            add_patch_property(patches, bug_id, mutant_id, 'label', label)
            add_patch_property(patches, bug_id, mutant_id, 'overlapping', False)
    
    with open(join(ssfix_dir, csv_file), newline='') as f:
        # lines = f.readlines()
        reader = csv.reader(f, delimiter=',')
        # for line in lines[1:]:
        for row in reader:
            project, id, mutant_id, line_in_buggy, line_in_patched, structural_score, conceptual_score, ssfix, correct = row
            if project == 'project': continue
            if correct == 'True': label = 'correct'
            else: label = 'overfitting'
            patch_dir = join(patch_root_dir, project, id, mutant_id)
            assert isdir(patch_dir), patch_dir
            bug_id = '-'.join([project, id])
            if isfile(join(patch_dir, 'CANT_FIX')): continue
            add_patch_property(patches, bug_id, mutant_id, 'ssfix', ssfix)

def display(dict_small, dict_merge, file_name):
    data = []
    text = []
    for bug_id in sorted(dict_merge):
        if not bug_id in dict_small.keys(): 
            data_tmp = np.nan
            text_tmp = '-'
        else: 
            data_tmp = dict_small[bug_id][0] / dict_small[bug_id][1]
            text_tmp = str(dict_small[bug_id][0]) + '/' + str(dict_small[bug_id][1])
        data.append([data_tmp, dict_merge[bug_id][0] / dict_merge[bug_id][1]])
        text.append([text_tmp, str(dict_merge[bug_id][0]) + '/' + str(dict_merge[bug_id][1])])
    
    df = pd.DataFrame(np.array(data), columns=['ASE patches', 'ASE + prapr patches'], index=sorted(dict_merge.keys()))
    fig = plt.figure(figsize=(8, 40))
    ax = fig.add_subplot(111)
    ax.axis('off')
    table = ax.table(cellText=text, rowLabels=df.index, colLabels=df.columns, loc='center', cellColours=plt.cm.Greys(df * 0.8))
    fig.savefig(file_name)

@DeprecationWarning
def print_ase_patches(ase_patches):
    bug_nums = [('Chart', 26), ('Closure', 133), ('Lang', 65), ('Math', 106), ('Time', 26)]
    count = 0
    for bug_tuple in bug_nums:
        project, num = bug_tuple
        for seq in range(num):
            id = str(seq + 1)
            bug_id = project + '-' + id
            if not bug_id in ase_patches: continue
            patches_dict = ase_patches[bug_id]
            for patch_name in patches_dict:
                tool, patch_id, Dlabel = patch_name.split('-')
                if patch_id == '0': sub_dir = 'Patches_ICSE'
                else: sub_dir = 'Patches_others'
                patch_dir = join(ASE_patch_dir, sub_dir, Dlabel, tool, project, id)
                if patch_id != '0': patch_dir = join(patch_dir, patch_id)
                if isfile(join(patch_dir, 'NOT_PLAUSIBLE')): continue
                count += 1
                assert isfile(join(patch_dir, 'src.patch'))
                print(join(patch_dir, 'src.patch'))
        
    assert count == 902
    sys.exit(0)
    
def average_correct_rank(rank_dict):
    rank_sum = 0
    for bug_id in rank_dict:
        rank, patch_num, tool = rank_dict[bug_id]
        rank_sum += rank
    
    return rank_sum / len(rank_dict)

def compare_correct_rank(rank_dict_small, rank_dict_merge):
    drop = 0
    for bug_id in rank_dict_small:
        rank_small, patch_num_small, tool = rank_dict_small[bug_id]
        rank_merge, patch_num_merge, tool = rank_dict_merge[bug_id]
        if rank_merge > rank_small: drop += 1
        
    print('%d out of %d bugs correct rank droped' % (drop, len(rank_dict_small)))
    
def get_balanced_dataset(prapr_ase_merged_patches, tool):
    correct_patches = list()
    ase_overfitting_patches = list()
    prapr_overfitting_patches = list()
    for bug_id in prapr_ase_merged_patches:
        patch_dict = prapr_ase_merged_patches[bug_id]
        for patch_name in patch_dict:
            patch_full_name = bug_id + '_' + patch_name
            properties = patch_dict[patch_name]
            label = properties['label']
            if label == 'correct': correct_patches.append((float(properties[tool]), label, patch_full_name))
            if label == 'overfitting': 
                if patch_name.startswith('mutant'): prapr_overfitting_patches.append((float(properties[tool]), label, patch_full_name))
                else: ase_overfitting_patches.append((float(properties[tool]), label, patch_full_name))
                
    correct_num = len(correct_patches)
    ase_overfitting_sample_num = round(584 / (584 + 1905) * correct_num)
    prapr_overfitting_sample_num = round(1905 / (584 + 1905) * correct_num)
    ase_overfitting_patches_sampled = random.sample(sorted(ase_overfitting_patches), ase_overfitting_sample_num)
    prapr_overfitting_patches_sampled = random.sample(sorted(prapr_overfitting_patches), prapr_overfitting_sample_num)
    assert len(ase_overfitting_patches_sampled + prapr_overfitting_patches_sampled) == correct_num, len(ase_overfitting_patches_sampled + prapr_overfitting_patches_sampled)
    return ase_overfitting_patches_sampled + prapr_overfitting_patches_sampled + correct_patches
    
if __name__ == '__main__':
    random.seed(1)
    ssfix_dir = '/home/junyang/PCC_repo/patch_correctness/RQ1/ssFix'
    s3_capgen_dir = '/home/junyang/PCC_repo/patch_correctness/RQ1/refined-scores/capgen_s3'
    opad_dir = '/home/junyang/PCC_repo/patch_correctness/RQ3/opad'
    ASE_patch_dir = '/home/junyang/PCC_repo/patch_correctness/ASE_Patches'
    prapr_patch_root_dir = '/home/junyang/PCC_repo/patch_correctness/prapr_src_patches_1.2'
    prapr_add_patch_root_dir = '/home/junyang/PCC_repo/patch_correctness/prapr_src_patches_2.0'
    dev_patch_root_dir = '/home/junyang/PCC_repo/patch_correctness/developer_patches_1.2'
    dev_add_patch_root_dir = '/home/junyang/PCC_repo/patch_correctness/developer_patches_2.0'
    prapr_csv = 'result_1.2.csv'
    prapr_add_csv = 'result_2.0.csv'
    dev_csv = 'result_dev_patches_1.2.csv'
    dev_add_csv = 'result_dev_patches_2.0.csv'
    
    ase_patches = dict()
    prapr_patches = dict()
    dev_patches = dict()
    prapr_add_patches = dict()
    tool = sys.argv[1]
    
    store_ASE_patches(ase_patches)
    store_prapr_or_dev_patches(prapr_patches, prapr_csv, prapr_patch_root_dir)
    store_prapr_or_dev_patches(prapr_add_patches, prapr_add_csv, prapr_add_patch_root_dir)
    # store_prapr_or_dev_patches(dev_patches, dev_csv, dev_patch_root_dir)
    # prapr_ase_merged_patches = merge_prapr_ase_patches(prapr_patches, ase_patches)
    prapr_new_patches = prapr_patches.copy()
    prapr_new_patches.update(prapr_add_patches)
    prapr_ase_merged_patches = merge_prapr_ase_patches(prapr_new_patches, ase_patches)
    # print(ase_patches["Math-59"])
    # print(prapr_ase_merged_patches["Math-59"])
    # print(sum([len(x) for x in prapr_new_patches.values()]))
    # print(sum([len(x) for x in prapr_patches.values()]))
    
    rank_ase_dict = rank_patches_per_bug(ase_patches, tool)
    rank_merged_dict = rank_patches_per_bug(prapr_ase_merged_patches, tool)
    # rank_dev_dict = rank_dev_patches(dev_patches, ase_patches, tool)
    # rank_dev_merged_dict = rank_dev_patches(dev_patches, prapr_ase_merged_patches, tool)
    
    # display(rank_ase_dict, rank_merged_dict, "/home/junyang/PCC_repo/patch_correctness/tables/" + tool + '-ASE-prapr-correct-rank.png')
    # display(rank_dev_dict, rank_dev_merged_dict, "/home/junyang/PCC_repo/patch_correctness/tables/" + tool + '-ASE-prapr-dev-rank.png')
    
    
    
    # compare_correct_rank(rank_ase_dict, rank_merged_dict)
    
    print('\nase patches:')
    print_confusion_matrix_from_patches(ase_patches, tool)
    print(average_correct_rank(rank_ase_dict))
    
    print('\nprapr 1.2 patches:')
    print_confusion_matrix_from_patches(prapr_patches, tool)
    print(average_correct_rank(rank_patches_per_bug(prapr_patches, tool)))
    
    print('\nprapr 2.0 patches:')
    print_confusion_matrix_from_patches(prapr_new_patches, tool)
    print(average_correct_rank(rank_patches_per_bug(prapr_new_patches, tool)))
    
    print('\nprapr + ase merged:')
    print_confusion_matrix_from_patches(prapr_ase_merged_patches, tool)
    print(average_correct_rank(rank_merged_dict))
    
    balanced_dataset = get_balanced_dataset(prapr_ase_merged_patches, tool)
    with open('/home/junyang/PCC_repo/patch_correctness/balanced_dataset/balanced_dataset_patches.txt', 'w') as f:
        f.writelines([x[-1] + '\n' for x in balanced_dataset])
    print('\nbalanced dataset:')
    print_confusion_matrix(balanced_dataset, int(len(balanced_dataset)/2), tool)
    