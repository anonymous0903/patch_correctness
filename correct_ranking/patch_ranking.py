import os, sys
from os.path import *
import csv, json, copy
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("agg")
from matplotlib import pyplot as plt


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

def merge_prapr_ase_patches(prapr_patches, ase_patches):
    merged_patches = copy.deepcopy(prapr_patches)
    ase_patches_filtered = dict()
    for bug_id in ase_patches:
        patches_dict = ase_patches[bug_id]
        for patch_name in patches_dict:
            patch_dict = patches_dict[patch_name]
            # We no longer exclude overlapping patches when merging
            # if patch_dict['overlapping']: continue
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

if __name__ == '__main__':
    ssfix_dir = '/home/junyang/PCC_repo/patch_correctness/RQ1/ssFix'
    s3_capgen_dir = '/home/junyang/PCC_repo/patch_correctness/RQ1/refined-scores/capgen_s3'
    opad_dir = '/home/junyang/PCC_repo/patch_correctness/RQ3/opad'
    ASE_patch_dir = '/home/junyang/PCC_repo/patch_correctness/ASE_Patches'
    prapr_patch_root_dir = '/home/junyang/PCC_repo/patch_correctness/prapr_src_patches_1.2'
    dev_patch_root_dir = '/home/junyang/PCC_repo/patch_correctness/developer_patches_1.2'
    prapr_csv = 'result_1.2.csv'
    dev_csv = 'result_dev_patches_1.2.csv'
    
    ase_patches = dict()
    prapr_patches = dict()
    dev_patches = dict()
    tool = sys.argv[1]
    
    store_ASE_patches(ase_patches)
    store_prapr_or_dev_patches(prapr_patches, prapr_csv, prapr_patch_root_dir)
    store_prapr_or_dev_patches(dev_patches, dev_csv, dev_patch_root_dir)
    prapr_ase_merged_patches = merge_prapr_ase_patches(prapr_patches, ase_patches)
    # print(ase_patches["Math-59"])
    # print(prapr_ase_merged_patches["Math-59"])
    
    rank_ase_dict = rank_patches_per_bug(ase_patches, tool)
    rank_merged_dict = rank_patches_per_bug(prapr_ase_merged_patches, tool)
    rank_dev_dict = rank_dev_patches(dev_patches, ase_patches, tool)
    rank_dev_merged_dict = rank_dev_patches(dev_patches, prapr_ase_merged_patches, tool)
    
    display(rank_ase_dict, rank_merged_dict, "/home/junyang/PCC_repo/patch_correctness/tables/" + tool + '-ASE-prapr-correct-rank.png')
    display(rank_dev_dict, rank_dev_merged_dict, "/home/junyang/PCC_repo/patch_correctness/tables/" + tool + '-ASE-prapr-dev-rank.png')