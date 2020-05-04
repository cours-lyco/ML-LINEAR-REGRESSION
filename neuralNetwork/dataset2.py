#!/usr/bin/env python3
import numpy as np 
import os
import argparse
import glob 

import sys
from collections  import OrderedDict
# Regex
import re
import matplotlib.pyplot as plt
# PArse xml
import xml.etree.ElementTree as ET

# Load / dump data
import pickle

from PIL import Image
from PIL import ImageDraw

from skimage.draw import line
from skimage.io import imread, imsave
import scipy.ndimage as ndimage
import pickle

import warnings
warnings.filterwarnings('ignore')


'''path of external device '''
rootDir = "/Volumes/Disk E/"
datasDir = rootDir +  "/crohme2016_inkml_datasets/CROHME2016_data/"

imagesDir = rootDir +  "/crohme2016_inkml_images/"
'''now get train valid test dir'''

trainDir = datasDir + 'Task-2-Symbols/task2-trainSymb2014/'
trainDirDatasets = trainDir + 'trainingSymbols/'
trainDir_ground_truth = trainDir + 'trainingSymbols/iso_GT.txt'
trainDatasImmagesDir = "/Volumes/Disk E/crohme2016_inkml_images/train_images"

testDir = datasDir + 'Task-2-Symbols/task2-testSymbols2014/'
testDir_grand_truth = testDir + "testSymbols_2016_iso_GT.txt"
testDirDatasets = testDir + "testSymbols/"
testDatasImmagesDir = "/Volumes/Disk E/crohme2016_inkml_images/test_images"

valDir = datasDir + 'Task-2-Symbols/task2-validation-isolatedTest2013b/'
valDirDatasets = datasDir + 'Task-2-Symbols/task2-validation-isolatedTest2013b/'
valDir_grand_truth = valDirDatasets + 'iso_GT.txt'
valDatasImmagesDir = "/Volumes/Disk E/crohme2016_inkml_images/val_images"


class InkDataExtractor(object):
    
    NS = {'ns': 'http://www.w3.org/2003/InkML',
          'xml': 'http://www.w3.org/XML/1998/namespace'}

    box_size = 90
    padding = 5
    #---------------------------------------------------------------------------------
    #                        CONSTRUCTOR
    #---------------------------------------------------------------------------------
    def __init__(self,  name):
        
        self.train_data = []
        self.test_data = []
        self.validation_data = []
        
        if name == 'get_train_data':
            self.train_labelsIds_labels, self.train_label_total = self.load_grand_truth(trainDir_ground_truth)
            self.read_all_inkml_file(trainDirDatasets)

        if name == 'get_test_data':
            self.test_labelsIds_labels, self.test_label_total = self.load_grand_truth(testDir_grand_truth)
            self.read_all_inkml_file(testDirDatasets)

        if name == 'get_val_data':
            self.val_labelsIds_labels, self.val_label_total = self.load_grand_truth(valDir_grand_truth)
            self.read_all_inkml_file(valDirDatasets)

        
    
    #-----------------------------------------------------------------------------------
    #
    #-----------------------------------------------------------------------------------
    def load_grand_truth(self, fileAbsPath):

        # Loads all categories that are available
        aDict = {}

        try:
            print("Loading grand_truth ............... ", fileAbsPath.split()[-2])
            with open(os.path.join(fileAbsPath), 'r') as desc:

                lines = desc.readlines()
        
                # Removing any whitespace characters appearing in the lines
                for line in lines:
                    split_data = line.split(",")
                    if split_data[1] == 'junk':
                        print("Junk")
                    else :
                        label_id, label = split_data[0], split_data[1].strip()
                        #print("loading label_id: {} with character: {}".format(label_id, label))
                        aDict[label_id] = label
                aDict = OrderedDict(sorted(aDict.items(), key=lambda t: t[0]))
                print("FINISH loading grand_truth ............... ",
                      fileAbsPath.split()[-2])
                print("Found: .............{} data".format(len(lines)))
                return aDict, len(lines)
        except FileNotFoundError as e:
            print(e)
            print("Oops!  grand truth file not found.  Try again...")
    #------------------------------------------------------------------------------------
    #                        EXTRACT FILE DATAS RETURN ArrayDict = [{label_id: [  [x1, y1], [x2, y2]   ]}]
    #------------------------------------------------------------------------------------

    def get_traces_data(self, inkml_file_abs_path):
        if inkml_file_abs_path.endswith('.inkml'):
            tree = ET.parse(inkml_file_abs_path)
            root = tree.getroot()
            doc_namespace = "{http://www.w3.org/2003/InkML}"
            'Stores traces_all with their corresponding id'
            traces_all_list = [{'label_id': trace_tag.get('id'),
                      'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                                    for axis_coord in coord[1:].split(' ')] if coord.startswith(' ') else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000) for axis_coord in coord.split(' ')]
                                    for coord in (trace_tag.text).replace('\n', '').split(',')]}
                                    for trace_tag in root.findall(doc_namespace + 'trace')]
            '''convert in dictionary traces_all  by id to make searching for references faster'''
            traces_all = {}
            for t in traces_all_list :
                traces_all[t["label_id"]] = t["coords"]
                #print("traces_alllalalalal",traces_all)
            #traces_all = OrderedDict(sorted(traces_all.items(), key=lambda t: int(t[0]) ))
            traces_all = {k: traces_all[k] for k in sorted(traces_all)}
            #traces_all.sort(key=lambda trace_dict: int(trace_dict['label_id']))
            #print(traces_all)
            return traces_all
        else:
            print('File ', inkml_file_abs_path, ' does not exist !')
            return {}

    
    #------------------------------------------------------------------------------------
    #                  MAKE   TRACE'S IMAGE : pattern_enc, pattern_corrupted = aDict = {features:image, label:... }, 
    #------------------------------------------------------------------------------------

    # trace_all contains coords only for 1 id

    def convert_to_imgs(self, traces_data, box_axis_size):
        
        pattern_drawn = np.ones(shape=(box_axis_size, box_axis_size), dtype=np.float32)
        # Special case of inkml file with zero trace (empty)
        
    
        'mid coords needed to shift the pattern'
        #print("traces_all['coords']"+str(traces_data))
        one_array_data = []
        for key, val in traces_data.items():
            for inner_val in val:
                one_array_data.append(inner_val)
        
        if len(one_array_data) == 0:
            print("array is nempty")
            return np.matrix(pattern_drawn * 255, np.uint8)
        
        min_x, min_y, max_x, max_y = self.get_min_coords(one_array_data)
        

        #min_x, min_y, max_x, max_y = self.get_min_coords([ trace for trace in [ traces for traces in traces_data ] ])
	    #print("min_x, min_y, max_x, max_y",min_x, min_y, max_x, max_y)
        'trace dimensions'
	    
        trace_height, trace_width = max_y - min_y, max_x - min_x
        if trace_height == 0:
            trace_height += 1
        if trace_width == 0:
            trace_width += 1
        '' 'KEEP original size ratio' ''
	
        trace_ratio = (trace_width) / (trace_height)
        box_ratio = box_axis_size / box_axis_size  # Wouldn't it always be 1
        scale_factor = 1.0
        
        '' 'Set \"rescale coefficient\" magnitude' ''
        if trace_ratio < box_ratio:
            scale_factor = ((box_axis_size-1) / trace_height)
        else:
            scale_factor = ((box_axis_size-1) / trace_width)
        #print("scale f : ", scale_factor)

        #for traces_all in one_array_data:
        'shift pattern to its relative position'
        shifted_trace = self.shift_trace(
                one_array_data, min_x=min_x, min_y=min_y)
       
        #print("shifted : "  , shifted_trace)
        'Interpolates a pattern so that it fits into a box with specified size'
        'method: LINEAR INTERPOLATION'
        
        try:
            scaled_trace = self.scaling(shifted_trace, scale_factor)
	        #print("inter : ", scaled_trace)
        
        except Exception as e:
            print(e)
            print('This data is corrupted - skipping.')
        
        'Get min, max coords once again in order to center scaled patter inside the box'
		#min_x, min_y, max_x, max_y = get_min_coords(interpolated_trace)
        
        centered_trace = self.center_pattern(scaled_trace, max_x=trace_width*scale_factor,
		                                max_y=trace_height*scale_factor, box_axis_size=box_axis_size-1)
        
        #print(" centered : " , centered_trace)
        'Center scaled pattern so it fits a box with specified size'
		
        pattern_drawn = self.draw_pattern(
		    centered_trace, pattern_drawn, self.box_size)
        
        #print("pattern size", pattern_drawn.shape)
        #print(np.matrix(pattern_drawn, np.uint8))
        return np.matrix(pattern_drawn * 255, np.uint8)


    #------------------------------------------------------------------------------------
    #                         
    #------------------------------------------------------------------------------------


    def read_all_inkml_file(self, data_dir_abs_path):
        '''Accumulates traces_data of all the inkml files\
        located in the specified directory'''
        patterns_enc = []
        classes_rejected = []

       
        'Check object is a directory'
        if os.path.isdir(data_dir_abs_path):
            #print(".... start loading  in folder: {}".format(data_dir_abs_path))
            for inkml_file in os.listdir(data_dir_abs_path):

                if inkml_file.endswith('.inkml'):
                    inkml_file_abs_path = os.path.join(
                        data_dir_abs_path, inkml_file)
                       
                    ''' **** Each entry in traces_data represent SEPARATE pattern\
                        which might(NOT) have its label encoded along with traces that it\'s made up of **** '''
                    traces_data_curr_inkml = self.get_traces_data(
                        inkml_file_abs_path)
                    fileName = inkml_file_abs_path.split('/')[-1]
                    print("{}  ..................>>> traces OK".format(fileName))
                    '''Each entry in patterns_enc is a dictionary consisting of \
                    pattern_drawn matrix and its label'''

                    image = self.convert_to_imgs(
                        traces_data_curr_inkml, self.box_size)

                    if self.padding > 0:
                        image = np.lib.pad(image, (self.padding, self.padding), 'constant', constant_values=255)
			
                    image = ndimage.gaussian_filter(image, sigma=(.5, .5), order=0)

                    img_basename = fileName.split('.')[0]
                    #print("'''''", data_dir_abs_path.split('/')[-2])
                    
                    if(data_dir_abs_path.split('/')[-2] == 'trainingSymbols'):
                        dir = trainDatasImmagesDir
                    elif (data_dir_abs_path.split('/')[-2] == 'testSymbols'):
                        dir = testDatasImmagesDir
                    elif (data_dir_abs_path.split('/')[-2] == 'task2-validation-isolatedTest2013b'):
                        dir = valDatasImmagesDir
                    #print(dir)
                    
                    try:
                        imsave( dir + os.sep + img_basename + '.png',image)
                    except Exception as e:
                        print(e)
                        print("Error: IMAGES SET DIR NOT FOUND")
                    #print(image)
                    print("{} <<<-------------------------- images OK".format(img_basename))

        #return patterns_enc

   

    #------------------------------------------------------------------------------------
    #                       UTILS
    #------------------------------------------------------------------------------------
    def get_min_coords(self, traces):

        min_x_coords = []
        min_y_coords = []
        max_x_coords = []
        max_y_coords = []
        
        x_coords = [coord[0] for coord in traces]
	    #print("xcoords"+str(x_coords))
        #a = [print(traces) for coord in traces]
        y_coords = [coord[1] for coord in traces]

    
        min_x_coords.append(min(x_coords))
        min_y_coords.append(min(y_coords))
        max_x_coords.append(max(x_coords))
        max_y_coords.append(max(y_coords))
           
        return min(min_x_coords), min(min_y_coords), max(max_x_coords), max(max_y_coords)
    
    'shift pattern to its relative position'


    def shift_trace(self, traces, min_x, min_y):
        
        shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in traces]
        return shifted_trace


    'Scaling: Interpolates a pattern so that it fits into a box with specified size'


    def scaling(self, traces, scale_factor=1.0):
    
        interpolated_trace = []
        'coordinate convertion to int type necessary'
	
        interpolated_trace = [[round(coord[0] * scale_factor),
                        round(coord[1] * scale_factor)] for coord in traces]
        return interpolated_trace


    def center_pattern(self, traces, max_x, max_y, box_axis_size):
        x_margin = int((box_axis_size - max_x) / 2)
        y_margin = int((box_axis_size - max_y) / 2)
        return self.shift_trace(traces, min_x=-x_margin, min_y=-y_margin)

    #------------------------------------------------------------------------------------
    #                        DRAW TRACES IMAGES
    #------------------------------------------------------------------------------------
    def draw_pattern(self,traces, pattern_drawn, box_axis_size):

        ' SINGLE POINT TO DRAW '
        if len(traces) == 1:
            print("Size is one")
            x_coord = traces[0][0]
            y_coord = traces[0][1]
            pattern_drawn[y_coord, x_coord] = 0.0
           

        else:
            ' TRACE HAS MORE THAN 1 POINT '

            'Iterate through list of traces endpoints'
            for pt_idx in range(len(traces) - 1):
                '''Indices of pixels that belong to the line. May be used to directly index into an array'''
                # pattern_drawn[line(r0=trace[pt_idx][1], c0=trace[pt_idx][0],
                #r1=trace[pt_idx + 1][1], c1=trace[pt_idx + 1][0])] = 0.0

                img = Image.fromarray(pattern_drawn)
                draw = ImageDraw.Draw(img)
                draw.line([(traces[pt_idx][0], traces[pt_idx][1]),
                           (traces[pt_idx + 1][0], traces[pt_idx + 1][1])], fill=0, width=3)

                pattern_drawn = np.array(img)
        return pattern_drawn
       



if __name__ == '__main__':

    data = None
    if len(sys.argv) !=  2:
        print("[USAGE ...........]:\n    ./fileName get_train_data \nOR   ./fileName get_test_data  \nOR  ./fileName get_val_data")
    else :
        if(sys.argv[1] == 'get_train_data' or sys.argv[1] == 'get_test_data' or sys.argv[1] == 'get_val_data'):
              data = InkDataExtractor(sys.argv[1])
