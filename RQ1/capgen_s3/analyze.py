import csv
import sys
import numpy as np
import os
from sklearn.metrics import roc_auc_score
import random

# please use python3 instead of python2

def get_top_N(sorted_dataset, N, reverse):
    # the dataset has been sorted to descending order for capgen and ascending order for s3
    top_N = list()
    others = list()
    tied = list()
    threshold = sorted_dataset[N - 1][0]
    for data in sorted_dataset:
        if data[0] > threshold: 
            if reverse: top_N.append(data)
            else: others.append(data)
        if data[0] < threshold: 
            if reverse: others.append(data)
            else: top_N.append(data)
        if data[0] == threshold: tied.append(data)
    
    if len(top_N) < N and len(top_N) + len(tied) > N:
        sampled = random.sample(tied, N - len(top_N))
        top_N = top_N + sampled
        for data in sampled:
            tied.remove(data)
        others = others + tied
        assert len(others) == len(sorted_dataset) - N
    elif len(top_N) + len(tied) == N:
        top_N = top_N + tied
    else: assert False
    return top_N, others

def get_full_ASE_scores(score_file):
    scores = []
    with open(score_file) as f:
        lines = f.readlines()
    for line in lines[1:]:
        patch_id, s3, capgen = line.strip().split('\t')
        if patch_id.find('Dcorrect') != -1: label = True
        else: label = False
        if sys.argv[1] == 's3':
            target = s3
        if sys.argv[1] == 'capgen':
            target = capgen
        if str(target).lower() != 'nan': scores.append((float(target), label))
    
    return scores

def ASE_patch_to_info(patch):
    assert patch.startswith('ASE_Patches')
    if patch.find('Patches_ICSE') != -1:
        label, tool, project, id = patch.split('/')[-4:]
        mutant_id = '0'
    elif patch.find('Patches_others') != -1:
        label, tool, project, id, mutant_id = patch.split('/')[-5:]
    else: assert False
    label = label[1:]

    return label, tool, project, id, mutant_id

def file_to_lines(file):
    items = list()
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        items.append(line.strip())
    
    return items

def look_up_score_ASE(tool, project, id, mutant_id, score_file):
    suffix = '.'.join([tool, project, id])
    if mutant_id != '0':
        suffix += ('.' + mutant_id)
    with open(score_file) as f:
        lines = f.readlines()
    for line in lines:
        patch_id, s3, capgen = line.strip().split('\t')
        if sys.argv[1] == 's3':
            target = s3
        if sys.argv[1] == 'capgen':
            target = capgen
        if patch_id.endswith(suffix): return float(target)
    
    assert False, '-'.join([tool, project, id, mutant_id])

def get_ASE_scores(correct_file, overfit_file, score_file):
    correct_scores, overfit_scores = [], []
    correct_patches_paths = file_to_lines(correct_file)
    overfit_patches_paths = file_to_lines(overfit_file)
    for patch in correct_patches_paths:
        if not patch.startswith('ASE_Patches'): continue
        label, tool, project, id, mutant_id = ASE_patch_to_info(patch)
        assert label == 'correct'
        score = look_up_score_ASE(tool, project, id, mutant_id, score_file)
        # print(score)
        if str(score).lower() == 'nan': continue
        correct_scores.append(score)
    for patch in overfit_patches_paths:
        if not patch.startswith('ASE_Patches'): continue
        label, tool, project, id, mutant_id = ASE_patch_to_info(patch)
        assert label == 'overfitting'
        score = look_up_score_ASE(tool, project, id, mutant_id, score_file)
        # print(score)
        if str(score).lower() == 'nan': continue
        overfit_scores.append(score)

    return correct_scores, overfit_scores

def check_patch(file, project, id, mutant_id):
    if mutant_id == 'mutant-0': return True
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

def AUC_score(merged_dataset):
    fact = []
    scores = []
    # the dataset follows a descending order, i.e., correct first, overfitting behind
    for patch in merged_dataset:
        if patch[1]:
            fact.append(1)
        else: fact.append(0)
        if sys.argv[1] == 'capgen':
            scores.append(patch[0])
        if sys.argv[1] == 's3':
            scores.append(0 - patch[0])
    
    fact = np.array(fact)
    scores = np.array(scores)

    return roc_auc_score(fact, scores)

def get_patches(file):
    with open(file) as f:
        reader = csv.reader(f)
        correct_patches = list()
        overfit_patches = list()
        count = 0
        for row in reader:
            count += 1
            if count == 1: continue
            project, id, mutant_id = row[:3]
            if not check_patch(file, project, id, mutant_id): continue
            toolASTDifferencing, toolCosine, toolStringDistance, toolVariable, toolSyntax, toolSemantic, s3tool, capgentool, correct = row[-9:]
            if sys.argv[1] == 's3':
                target = s3tool
            if sys.argv[1] == 'capgen':
                target = capgentool
            if correct == 'null': continue
            if correct == 'TRUE':
                correct_patches.append(float(target))
            if correct == 'FALSE':
                overfit_patches.append(float(target))
    
    return correct_patches, overfit_patches

def scores_to_file(scores, file):
    with open(file, 'w') as f:
        for score in scores:
            f.write(score + '\n')

def get_balanced_prapr_overfit_patches(result_file, balanced_overfit_patches_file):
    if result_file.endswith('result_1.2.csv'): prefix = 'prapr_src_patches_1.2/'
    if result_file.endswith('result_2.0.csv'): prefix = 'prapr_src_patches_2.0/'
    scores = []
    result_lines = file_to_lines(result_file)
    balanced_overfit_patches = file_to_lines(balanced_overfit_patches_file)
    for patch in balanced_overfit_patches:
        if patch.startswith(prefix):
            project, id, mutant_id = patch.split('/')[-3:]
            found = False
            count = 0
            for line in result_lines:
                if line.startswith(','.join([project, id, mutant_id]) + ','): 
                    s3, capgen = line.split(',')[-3 : -1]
                    found = True
                    if line.split(',')[-1] != 'null':
                        if sys.argv[1] == 's3':
                            target = s3
                        if sys.argv[1] == 'capgen':
                            target = capgen
                        scores.append(float(target))
                        count += 1
                        assert line.split(',')[-1] == 'FALSE'
                    # else: scores.append('nan')
            assert found
            assert count <= 1
    return scores



if __name__ == '__main__':
    random.seed(1)
    tool = sys.argv[1]
    dataset = sys.argv[2]
    assert tool == 'capgen' or 's3', 'invalid input!'
    assert dataset in ['1.2', '2.0', 'merge', 'balance'], 'invalid input!'
    patch_root_dir_1 = '../../prapr_src_patches_1.2'
    patch_root_dir_2 = '../../prapr_src_patches_2.0'
    ASE_scores_file = 'ASE_patch_score_capgen_s3.txt'
    all_overfit_patches_file = '../../balanced_dataset/overfitting_patches_all.txt'
    all_correct_patches_file = '../../balanced_dataset/correct_patches_all.txt'
    balanced_overfit_patches_file = '../../balanced_dataset/overfitting_patches_balanced.txt'

    # 1.2 dataset
    prapr_correct_patches_1, prapr_overfit_patches_1 = get_patches('result_1.2.csv')
    prapr_balanced_overfit_patches = get_balanced_prapr_overfit_patches('result_1.2.csv', balanced_overfit_patches_file)
    dev_correct_patches_1, dev_overfit_patches_1 = get_patches('result_dev_patches_1.2.csv')

    # 2.0 dataset
    prapr_correct_patches_2, prapr_overfit_patches_2 = get_patches('result_2.0.csv')
    prapr_balanced_overfit_patches_2 = get_balanced_prapr_overfit_patches('result_2.0.csv', balanced_overfit_patches_file)
    dev_correct_patches_2, dev_overfit_patches_2 = get_patches('result_dev_patches_2.0.csv')

    # ASE patches
    ASE_correct_patches, ASE_overfit_patches = get_ASE_scores(all_correct_patches_file, all_overfit_patches_file, ASE_scores_file)
    ASE_correct_patches, ASE_balanced_overfit_patches = get_ASE_scores(all_correct_patches_file, balanced_overfit_patches_file, ASE_scores_file)

    # balanced patches
    print('balanced patches: ')
    print(len(prapr_balanced_overfit_patches))
    print(len(prapr_balanced_overfit_patches_2))
    balanced_overfit_patches = prapr_balanced_overfit_patches + prapr_balanced_overfit_patches_2 + ASE_balanced_overfit_patches
    print(len(balanced_overfit_patches))
    print(np.mean(balanced_overfit_patches))


    print('prapr patches')
    print(len(prapr_correct_patches_1 + prapr_correct_patches_2))
    print(len(prapr_overfit_patches_1 + prapr_overfit_patches_2))
    print('dev patches')
    print(len(dev_correct_patches_1 + dev_correct_patches_2))
    print(len(dev_overfit_patches_1 + dev_overfit_patches_2))

    # scores of prapr, ase, dev patches
    print(np.mean(prapr_correct_patches_1 + prapr_correct_patches_2))
    print(np.mean(prapr_overfit_patches_1 + prapr_overfit_patches_2))
    print(np.mean(ASE_correct_patches))
    print(np.mean(ASE_overfit_patches))
    print(np.mean(dev_correct_patches_1 + dev_correct_patches_2))

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

    print('correct and overfit patches')
    print(len(correct_patches))
    print(len(overfit_patches))

    merged_dataset = merge_two_group(correct_patches, overfit_patches)
    if sys.argv[1] == 'capgen': reverse = True
    if sys.argv[1] == 's3': reverse = False
    sorted_dataset = sorted(merged_dataset, key=lambda x: x[0], reverse=reverse)

    print('AUC score')
    print(AUC_score(merged_dataset))

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    
    # get top N patches
    top_N, others = get_top_N(sorted_dataset, len(correct_patches), reverse)
    # print(top_N)
    for element in top_N:
        if element[1]:
            TN += 1
        else:
            FN += 1
    for element in others:
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