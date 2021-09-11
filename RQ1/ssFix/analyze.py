import csv
from os.path import join
import numpy as np
import random
import os,sys
from sklearn.metrics import roc_auc_score


def get_full_ASE_scores(score_file):
    scores = []
    lines = file_to_lines(score_file)
    for line in lines[1:]:
        correct = line.split(',')[-1] == 'True'
        score = float(line.split(',')[-2])
        scores.append((score, correct))

    return scores

def AUC_score(merged_dataset):
    fact = []
    scores = []
    # the dataset follows a descending order, i.e., correct first, overfitting behind
    for patch in merged_dataset:
        if patch[1]:
            fact.append(1)
        else: fact.append(0)
        scores.append(patch[0])
    
    fact = np.array(fact)
    scores = np.array(scores)

    return roc_auc_score(fact, scores)

def look_up_score(result_file, tool, project, id, mutant_id):
    if file_to_lines(result_file)[0].startswith('tool'): prapr_patch = False
    else: prapr_patch = True
    for line in file_to_lines(result_file):
        if prapr_patch:
            if [project, id, mutant_id] == line.split(',')[:3]:
                return line.split(',')[-2]
        else:
            if [tool, project, id, mutant_id] == line.split(',')[:4]:
                return line.split(',')[-2]
    
    assert False, [result_file, tool, project, id, mutant_id]

def file_to_lines(file):
    items = list()
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        items.append(line.strip())
    
    return items

def get_balanced_overfit_patches(prapr_result_file, prapr_result_file_2, ASE_result_file, balanced_overfit_file_path):
    balanced_overfit_patches = []
    balanced_overfit_patch_paths = file_to_lines(balanced_overfit_file_path)
    for overfit_patch in balanced_overfit_patch_paths:
        if overfit_patch.startswith('prapr_src_patches_1.2'):
            # this is a prapr patch, 1.2 
            project, id, mutant_id = overfit_patch.split('/')[-3:]
            tool = 0
            score = look_up_score(prapr_result_file, tool, project, id, mutant_id)
        elif overfit_patch.startswith('prapr_src_patches_2.0'):
            # this is a prapr patch, 2.0
            project, id, mutant_id = overfit_patch.split('/')[-3:]
            tool = 0
            score = look_up_score(prapr_result_file_2, tool, project, id, mutant_id)
        elif overfit_patch.startswith('ASE_Patches'):
            # this is a patch from ASE paper
            if overfit_patch.split('/')[1] == 'Patches_ICSE':
                tool, project, id = overfit_patch.split('/')[-3:]
                mutant_id = '0'
            if overfit_patch.split('/')[1] == 'Patches_others':
                tool, project, id, mutant_id = overfit_patch.split('/')[-4:]
            score = look_up_score(ASE_result_file, tool, project, id, mutant_id)
        else: assert False
        balanced_overfit_patches.append(float(score))

    return balanced_overfit_patches
        

def get_patches(result_file):
    correct_patches = []
    overfit_patches = []
    with open(result_file) as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            count += 1
            if count == 1: continue
            project, id, mutant_id = row[:3]
            lines = file_to_lines(result_file)
            if lines[0].startswith('tool'): project, id, mutant_id = row[1:4]
            if not check_patch(result_file, project, id, mutant_id): continue
            structural_scores, conceptual_scores, sum, correct = row[-4:]
            target = sum
            if correct == 'True':
                correct_patches.append(float(target))
            if correct == 'False':
                overfit_patches.append(float(target))
    return correct_patches, overfit_patches

def check_patch(file, project, id, mutant_id):
    if mutant_id == 'mutant-0': return True
    if file.endswith('ASE_patches.csv'): return True
    if file.endswith('1.2.csv'): root_dir = patch_root_dir_1
    if file.endswith('2.0.csv'): root_dir = patch_root_dir_2
    mutant_dir = os.path.join(root_dir, project, id, mutant_id)
    assert os.path.isdir(mutant_dir), mutant_dir
    if os.path.isfile(os.path.join(mutant_dir, 'CANT_FIX')): 
        return False
    if os.path.isfile(os.path.join(mutant_dir, 'NO_DIFF')): 
        return False

    return True

def merge_two_group(correct_patches, overfit_patches):
    merged_patches = list()
    for patch in correct_patches:
        merged_patches.append((patch, True))
    for patch in overfit_patches:
        merged_patches.append((patch, False))
    
    return merged_patches


if __name__ == '__main__':
    dataset = sys.argv[1]
    assert dataset in ['1.2', '2.0', 'merge', 'balance'], 'invalid input!'
    patch_root_dir_1 = '../../prapr_src_patches_1.2'
    patch_root_dir_2 = '../../prapr_src_patches_2.0'
    balanced_overfit_file_path = '../../balanced_dataset/overfitting_patches_balanced.txt'
    correct_patches = list()
    overfit_patches = list()
    # 1.2 dataset
    prapr_correct_patches_1, prapr_overfit_patches_1 = get_patches('result_1.2.csv')
    dev_correct_patches, dev_overfit_patches = get_patches('result_dev_patches_1.2.csv')
    print('prapr 1.2 correct patches: ' + str(len(prapr_correct_patches_1)))
    # 2.0 dataset
    prapr_correct_patches_2, prapr_overfit_patches_2 = get_patches('result_2.0.csv')
    dev_correct_patches_2, dev_overfit_patches_2 = get_patches('result_dev_patches_2.0.csv')
    print('prapr 2.0 correct patches: ' + str(len(prapr_correct_patches_2)))
    # ASE dataset
    balanced_overfit_patches = get_balanced_overfit_patches('result_1.2.csv', 'result_2.0.csv', 'result_ASE_patches.csv', balanced_overfit_file_path)
    print('balanced overfit patches' + str(len(balanced_overfit_patches)))
    ASE_correct_patches, ASE_overfit_patches = get_patches('result_ASE_patches.csv')
    print('ASE correct patches: ' + str(len(ASE_correct_patches)))
    print('ASE overfitting patches: ' + str(len(ASE_overfit_patches)))

    print('prapr and dev patches score:')
    print(np.mean(prapr_correct_patches_1 + prapr_correct_patches_2))
    print(np.mean(prapr_overfit_patches_1 + prapr_overfit_patches_2))
    print(np.mean(ASE_correct_patches))
    print(np.mean(ASE_overfit_patches))
    print(np.mean(dev_correct_patches + dev_correct_patches_2))
    
    if dataset == '1.2':
        correct_patches = prapr_correct_patches_1
        overfit_patches = prapr_overfit_patches_1

    if dataset == '2.0':
        correct_patches = prapr_correct_patches_1 + prapr_correct_patches_2
        overfit_patches = prapr_overfit_patches_1 + prapr_overfit_patches_2

    if dataset == 'merge':
        correct_patches = prapr_correct_patches_1 + prapr_correct_patches_2 + ASE_correct_patches
        overfit_patches = prapr_overfit_patches_1 + prapr_overfit_patches_2 + ASE_overfit_patches

    if dataset == 'balance':
        correct_patches = prapr_correct_patches_1 + prapr_correct_patches_2 + ASE_correct_patches
        overfit_patches = balanced_overfit_patches

    print(len(correct_patches))
    print(len(overfit_patches))

    print(np.mean(correct_patches))
    print(np.mean(overfit_patches))

    merged_dataset = merge_two_group(correct_patches, overfit_patches)
    sorted_dataset = sorted(merged_dataset, key=lambda x: x[0], reverse=True)

    print('AUC score')
    print(AUC_score(merged_dataset))

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for element in sorted_dataset[:len(correct_patches)]:
        if element[1]:
            TN += 1
        else:
            FN += 1
    for element in sorted_dataset[len(correct_patches):]:
        if element[1]:
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