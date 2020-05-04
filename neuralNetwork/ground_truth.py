#!/usr/bin/env python3
import sys, os
from collections import OrderedDict
import numpy as np 

'''path of external device '''
rootDir = "/Volumes/Disk E/"
datasDir = rootDir + "/crohme2016_inkml_datasets/CROHME2016_data/"

imagesDir = rootDir + "/crohme2016_inkml_images/"
'''now get train valid test dir'''

trainDir = datasDir + 'Task-2-Symbols/task2-trainSymb2014/'
trainDirDatasets = trainDir + 'trainingSymbols/'
trainDir_ground_truth = trainDir + 'trainingSymbols/iso_GT.txt'
trainDatasImmagesDir = "/Volumes/Disk E/crohme2016_inkml_images/train_images"

testDir = datasDir + 'Task-2-Symbols/task2-testSymbols2014/'
testDir_ground_truth = testDir + "testSymbols_2016_iso_GT.txt"
testDirDatasets = testDir + "testSymbols/"
testDatasImmagesDir = "/Volumes/Disk E/crohme2016_inkml_images/test_images"

valDir = datasDir + 'Task-2-Symbols/task2-validation-isolatedTest2013b/'
valDirDatasets = datasDir + 'Task-2-Symbols/task2-validation-isolatedTest2013b/'
valDir_ground_truth = valDirDatasets + 'iso_GT.txt'
valDatasImmagesDir = "/Volumes/Disk E/crohme2016_inkml_images/val_images"


class GT(object):
   #trains total characters: 85802
   #tests total characters: 18435
   #val total characters: 12504
   '''AFTER REMOVING JUNK FILES'''
   #self.y_categorical['train_y'] : (85802, 101)
   #self.y_categorical['test_y'] : (10019, 101)
   #self.y_categorical['val_y'] : (6083, 101)

   #INFOS:
      #les deux varibales utils ici: self.gt et self.y_categorical
   def __init__(self):  # name == get_train_data OR name == get_test_data OR name == get_val_data
        '''  self.gtruth['train_gt'] : {  '(': [0, 7, 23, 40, ..],  '+': []    '''
        '''  self.gtruth['test_gt'] : {  '(': [22, 89, 413, 340, ..],  '+': []    '''
        '''  self.gtruth['val_gt'] : {  '(': [145, 78, 56, 89, ..],  '+': []    '''

        self.gtruth = {'train_gt': [], 'test_gt': [], 'val_gt': []}

        ''' self.y_categorical['train_y'] = matrix_1x101 avec 1=index courant et zero=le reste'''
        self.y_categorical = {'train_y': [], 'test_y': [], 'val_y': []}

        self.get_all_sorted_categorical()

        self.y_categorical['train_y'] = np.array(
           self.y_categorical['train_y'], dtype=np.float32)

        self.y_categorical['test_y'] = np.array(
            self.y_categorical['test_y'], dtype=np.float32)

        self.y_categorical['val_y'] = np.array(
            self.y_categorical['val_y'], dtype=np.float32)

        #a = self.get_label_value_in_categorical_array('X', self.sortedDict, arr)

        #---------------------------------------------------------------------------------------------------#
        #  Result: y_train = Matrix 101 x 101, y_test = Matrix 101 x 101, y_val = Matrix 101 x 101
        #---------------------------------------------------------------------------------------------------#

   def get_all_sorted_categorical(self):
        gt_directories_files = [trainDir_ground_truth, testDir_ground_truth, valDir_ground_truth]
        imagesDir = [trainDatasImmagesDir, testDatasImmagesDir, valDatasImmagesDir]

        for index, abs_gt_file_path in enumerate(gt_directories_files):
            if index == 0:
                print("-------->>>>> Start reading TRAIN grount truth ")
            if index == 1:
                print("-------->>>>> Start reading TEST grount truth ")
            if index == 2:
                print("-------->>>>> Start reading VAL grount truth ")

            sorted_gt, total_character, y = self.read_gt(
                abs_gt_file_path, imagesDir[index])

            if index == 0:
                print("......... Finish reading [TRAIN] grount truth: {} characters .......\n".format(
                    total_character))
            if index == 1:
                print("..........Finish reading [TEST] grount truth {} characters ..........\n".format(
                    total_character))
            if index == 2:
                print("..........Finish reading [VAL] grount truth {} characters ...........\n".format(
                    total_character))

            length = len(sorted_gt.keys())

            for i, el in enumerate(y):
                x = np.zeros(length, dtype=np.float32)
                x[el] = 1.0

                if index == 0:
                    self.y_categorical['train_y'].append(x)
                    self.gtruth['train_gt'] = sorted_gt
                if index == 1:
                    self.y_categorical['test_y'].append(x)
                    self.gtruth['test_gt'] = sorted_gt
                if index == 2:
                    self.y_categorical['val_y'].append(x)
                    self.gtruth['val_gt'] = sorted_gt

   def read_gt(self, abs_gt_file_path, imagesDir):
        '''Loads the routes of the png files'''
        try:
            #fileName = os.path.basename(abs_gt_file_path)
            #print("reading ground truth {} ..................".format(fileName))
            with open(abs_gt_file_path) as file:
                lines = file.readlines()

                aDict = {}
                aLabelsSet = []
                total_character = 0
                all_labels = []

                for index, line in enumerate(lines):
                    line = line.strip()

                    parts = line.split(',')
                    label = parts[1].strip()

                    data = parts[0].split('_')
                    index = data[-1].strip()
                    if(label != 'junk'):
                        all_labels.append(label)
                    if label not in aLabelsSet and label != 'junk':
                        aLabelsSet.append(label)

                    if label in aDict and label != 'junk':
                        #aDict[label].append(imagesDir + '/' + 'iso' + index + '.png')
                        aDict[label].append(int(index))
                    elif label != 'junk':
                        #aDict[label] = [imagesDir + '/' + 'iso' + index + '.png']
                        aDict[label] = [int(index)]
                    total_character = total_character + 1

            #sort dictionary but others method doesn't work
            bDict = {}
            y = []
            for label in aLabelsSet:
                bDict[label] = aDict[label]
            for label in all_labels:
                for i, el in enumerate(aLabelsSet):
                    if(el == label):
                        y.append(i)

            #aDict = OrderedDict(sorted(aDict.items(), key=lambda t: t[0]))
            # print("dictionnary sorted keys : ", bDict.keys())
            #print("Finish readin ground {} truth: {}".format(fileName, len(bDict)))
            #print("bDict length:........", len(bDict))

            return bDict, total_character, y

        except FileNotFoundError as e:
            print(e)


if __name__ == '__main__':
      gt = GT()
      print("self.gtruth['train_gt'] : {}\n".format(len(gt.gtruth['train_gt'])))
      print("self.gtruth['test_gt'] : {}\n".format(len(gt.gtruth['test_gt'])))
      print("self.gtruth['val_gt'] : {}\n".format(len(gt.gtruth['val_gt'])))

      print("self.y_categorical['train_y'] : {}\n".format(gt.y_categorical['train_y'].shape))
      print("self.y_categorical['test_y'] : {}\n".format(gt.y_categorical['test_y'].shape))
      print("self.y_categorical['val_y'] : {}\n".format(gt.y_categorical['val_y'].shape))
