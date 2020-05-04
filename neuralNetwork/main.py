#!/usr/bin/env python3
import sys,  os
import matplotlib.pyplot as plt 
#import cv2 as cv

'-------------------------------------------------------'
'               MODULE IMPORT                           '            
'-------------------------------------------------------'
from dataset import *

'-------------------------------------------------------'
'               EXTERNAL PATHS                          '
'-------------------------------------------------------'
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


'--------------------------------------------------------'

if __name__ == '__main__':

    
       
    print('---------[USAGE TO SET DATAS]: --------------\n./main.py get_train_data \nOR ./main.py get_test_data \nOR ./main.py get_val_data')
    print('OR TO TRAIN MODEL: ./main get_train_data')
    print('OR TO TEST MODEL: ./main get_test_data')
    print('OR TO VALIDATE MODEL: ./main get_val_data')
   
    print('---------[USAGE TO CHECK DATAS IMAGES]:\n')
    print("OR TO SHOW TRAIN IMAGE: ./main get_train_image number")
    print("OR TO SHOW TEST IMAGE: ./main get_test_image number")
    print("OR TO SHOW VAL IMAGE: ./main get_val_image number")
    
    if(len(sys.argv) == 2):
        data = InkDataExtractor(sys.argv[1])
    if(len(sys.argv) == 3):
        index = sys.argv[2]
        if(sys.argv[1] == 'get_train_image'): 
            path = trainDatasImmagesDir + os.sep + 'iso' + index + '.png'
        elif(sys.argv[1] == 'get_test_image'):
            path = testDatasImmagesDir + os.sep + 'iso' + index + '.png'
        elif(sys.argv[1] == 'get_val_image'):
            path = valDatasImmagesDir + os.sep + 'BOTH' + index + '.png'
        #img = cv.imread(path)
        #plt.imshow(img)
    else:
        print("Please: check Usage\n");  
   
