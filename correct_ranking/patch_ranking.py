import os
from os.path import *
import csv, json

# get scores of ssfix, s3, capgen, opad

def add_patch_property(bugs_dict, bug_id, patch_name, property_name, property_value):
    if not bug_id in bugs_dict.keys(): bugs_dict[bug_id] = dict()
    patches_dict = bugs_dict[bug_id]
    if not patch_name in patches_dict.keys(): patches_dict[patch_name] = dict()
    patch_dict = patches_dict[patch_name]
    patch_dict[property_name] = property_value

def rank_correct_patch(patches_dict, tool):
    # the patches should contain at least one correct patch and one overfitting patch
    
    correct_patches_dict = {k: v for k, v in patches_dict.items() if v['label'] == 'correct'}
    overfitting_patches_dict = {k: v for k, v in patches_dict.items() if v['label'] == 'overfitting'}
    # print(str(len(correct_patches_dict)) + '\t' + str(len(overfitting_patches_dict)))
    if len(correct_patches_dict) > 0 and len(overfitting_patches_dict) == 0: return 1
    if len(correct_patches_dict) ==0 and len(overfitting_patches_dict) > 0: return None
    # correct_patch_score = max(correct_patches_dict, key=lambda x: float(correct_patches_dict[x][tool]))
    correct_patch_score = max([float(patch_dict[tool]) for patch_dict in correct_patches_dict.values()])
    rank = 1
    # print('length: ' + str(len(patches_dict)))
    for patch_name in patches_dict.keys():
        score = float(patches_dict[patch_name][tool])
        if score > correct_patch_score: rank += 1
    
    # print(rank)
    return rank
     
def rank_patches_per_bug(bugs_dict):
    tool = 'capgen'
    for bug_id in bugs_dict:
        patches_dict = bugs_dict[bug_id]
        rank = rank_correct_patch(patches_dict, tool)
        print(bug_id + ': ' + str(rank) + ' out of ' + str(len(patches_dict)))
   
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
            if isfile(join(patch_dir, 'NOT_PLAUSIBLE')): continue
            if isfile(join(patch_dir, 'MISLABEL')):
                if label == 'correct': label = 'overfitting'
                elif label == 'overfitting': label = 'correct'
            bug_id = '-'.join([project, id])
            patch_name = '-'.join([tool, patch_id, 'D' + label])
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
            if isfile(join(patch_dir, 'NOT_PLAUSIBLE')): continue
            if isfile(join(patch_dir, 'MISLABEL')):
                if label == 'correct': 
                    label = 'overfitting'
                elif label == 'overfitting': label = 'correct'
            bug_id = '-'.join([project, id])
            patch_name = '-'.join([tool, patch_id, 'D' + label])
            add_patch_property(ase_patches, bug_id, patch_name, 'ssfix', ssfix)
            assert ase_patches[bug_id][patch_name]['label'] == label, ase_patches[bug_id]

if __name__ == '__main__':
    ssfix_dir = '/home/junyang/PCC_repo/patch_correctness/RQ1/ssFix'
    s3_capgen_dir = '/home/junyang/PCC_repo/patch_correctness/RQ1/refined-scores/capgen_s3'
    opad_dir = '/home/junyang/PCC_repo/patch_correctness/RQ3/opad'
    prapr_csv = 'result_1.2.csv'
    dev_csv = 'result_dev_patches_1.2.csv'
    ase_patches = dict()
    
    ASE_patch_dir = '/home/junyang/PCC_repo/patch_correctness/ASE_Patches'
    store_ASE_patches(ase_patches)
    # print(len(ase_patches))
    rank_patches_per_bug(ase_patches)