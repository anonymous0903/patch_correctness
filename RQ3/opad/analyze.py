import os
import csv
import sys
from os.path import join, exists, isfile, isdir

def check_patch(project, id, mutant_id):
    if mutant_id == 'mutant-0': return True
    patch_dir = join(patch_root_dir, project, id, mutant_id)
    correct_path = join(patch_dir, 'correct')
    return isfile(correct_path)

if __name__ == '__main__':
    test_suite = sys.argv[1]
    dataset = sys.argv[2]
    assert dataset == '1.2' or dataset == '2.0'
    assert test_suite == 'evosuite' or test_suite == 'randoop'
    patch_root_dir = '../../prapr_src_patches_' + dataset
    result_file = test_suite + '_opad_result_' + dataset + '.csv'

    TP = TN = FN = FP = 0
    with open(result_file) as f:
        reader = csv.reader(f)
        for row in reader:
            [project, id, mutant_id, predict]  = row
            # if mutant_id == 'mutant-0': continue
            fact = check_patch(project, id, mutant_id)
            if predict == 'overfitting':
                if not fact:
                    TP += 1
                else:
                    FP += 1
            if predict == 'correct':
                if not fact:
                    FN += 1
                else:
                    TN += 1
    print('dataset: ' + dataset)
    print('test suite: ' + test_suite)
    print('TP: ' + str(TP))
    print('FP: ' + str(FP))
    print('TN: ' + str(TN))
    print('FN: ' + str(FN))
    precision = TP / float(TP + FP)
    recall = TP / float(TP + FN)
    f1 = 2 * precision * recall /(precision + recall)
    print('precision: ' + str(precision))
    print('recall: ' + str(recall))
    print("F1: " + str(f1))