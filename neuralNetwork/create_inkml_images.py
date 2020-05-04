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
testDir_ground_truth = testDir + "testSymbols_2016_iso_GT.txt"
testDirDatasets = testDir + "testSymbols/"
testDatasImmagesDir = "/Volumes/Disk E/crohme2016_inkml_images/test_images"

valDir = datasDir + 'Task-2-Symbols/task2-validation-isolatedTest2013b/'
valDirDatasets = datasDir + 'Task-2-Symbols/task2-validation-isolatedTest2013b/'
valDir_ground_truth = valDirDatasets + 'iso_GT.txt'
valDatasImmagesDir = "/Volumes/Disk E/crohme2016_inkml_images/val_images"

'''-------------------------------------------------------------- '''
from ground_truth import *


class InkDataExtractor(GT):
    
    NS = {'ns': 'http://www.w3.org/2003/InkML',
          'xml': 'http://www.w3.org/XML/1998/namespace'}

    box_size = 28
    padding = 0
    #---------------------------------------------------------------------------------
    #                        CONSTRUCTOR
    #---------------------------------------------------------------------------------
    def __init__(self,  name='image_already_set'):
        super().__init__()
        if name == 'train':
            self.dir = trainDirDatasets
            self.slice_start = 3
            self.slice_stop = -6
        elif name == 'test':
            self.dir = testDirDatasets
            self.slice_start = 3
            self.slice_stop = -6
        elif name == 'validation':
            self.dir = valDirDatasets
            self.slice_start = 4
            self.slice_stop = -6
        else:
            print("OK CHECK IMAGES IN YOUR EXTRA DEVICES . \n")
        self.read_all_inkml_file(self.dir)


    '-------------------------------------------------------------------------'

    def read_all_inkml_file(self, data_dir_abs_path):
        '''Accumulates traces_data of all the inkml files\
        located in the specified directory'''
    
        'Check object is a directory'
        if os.path.isdir(data_dir_abs_path):
            #print(".... start loading  in folder: {}".format(data_dir_abs_path))
            inkml_files = sorted(glob.glob(data_dir_abs_path + "*.inkml"))
            inkml_files = sorted(inkml_files, key=lambda name: int((os.path.basename(name))[self.slice_start : self.slice_stop]))
            
            
            #inkml_files = inkml_files.sort(key=self.sortGlobFilesArray)
            for  inkml_file_abs_path in inkml_files:
                
                #if inkml_file.endswith('.inkml'):
                
                ''' **** Each entry in traces_data represent SEPARATE pattern\
                which might(NOT) have its label encoded along with traces that it\'s made up of **** '''
                    
                aDict_traces = self.parse_inkml(inkml_file_abs_path)
                    
                traces_data_curr_inkml = self.get_traces_data( aDict_traces, id_set=None)
                
                img_basename = os.path.basename(inkml_file_abs_path)

                print("{}  ..................>>> traces OK".format(img_basename))
                '''Each entry in patterns_enc is a dictionary consisting of \
                pattern_drawn matrix and its label'''

                image = self.convert_to_imgs(
                        traces_data_curr_inkml, self.box_size)
                    
                if self.padding > 0:
                        image = np.lib.pad(
                            image, (self.padding, self.padding), 'constant', constant_values=255)

                image = ndimage.gaussian_filter(
                        image, sigma=(.5, .5), order=0)

                
                    #print("'''''", data_dir_abs_path.split('/')[-2])

                if(data_dir_abs_path.split('/')[-2] == 'trainingSymbols'):
                    dir = trainDatasImmagesDir
                elif (data_dir_abs_path.split('/')[-2] == 'testSymbols'):
                    dir = testDatasImmagesDir
                elif (data_dir_abs_path.split('/')[-2] == 'task2-validation-isolatedTest2013b'):
                    dir = valDatasImmagesDir
                    #print(dir)

                try:
                    file_no_extension = img_basename.split('.')[0]
                    imsave(dir + os.sep +  file_no_extension + '.png', image)
                except Exception as e:
                    print(e)
                    print("Error: IMAGES SET DIR NOT FOUND")
                #print(image)
                print("{} <<<-------------------------- images OK".format(img_basename))
                
        
    #--------------------------------------------------------------------------#
    #Exemple of output of parse_inkml with file iso4.inkml
    #{
    #   '8': [[538, 187], [534, 189], [546, 189], [548, 189], [551, 188]], 
    #   '7': [[536, 178], [534, 178], [533, 178], [534, 178], [536, 178], [550, 179], [551, 179], [553, 181], [552, 182], [551, 182], [549, 183]]
    #}

    def parse_inkml(self, inkml_file_abs_path):
        if inkml_file_abs_path.endswith('.inkml'):
            tree = ET.parse(inkml_file_abs_path)
            root = tree.getroot()
            doc_namespace = "{http://www.w3.org/2003/InkML}"
            'Stores traces_all with their corresponding id'
            
            traces_all_list = [{'id': trace_tag.get('id'),
                            'coords': [[round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                                        for axis_coord in coord[1:].split(' ')] if coord.startswith(' ')
                                       else [round(float(axis_coord)) if float(axis_coord).is_integer() else round(float(axis_coord) * 10000)
                                             for axis_coord in coord.split(' ')]
                                       for coord in (trace_tag.text).replace('\n', '').split(',')]}
                           for trace_tag in root.findall(doc_namespace + 'trace')]

            'convert in dictionary traces_all  by id to make searching for references faster'
            traces_all = {}
            
            for t in traces_all_list:
                traces_all[t["id"]] = t["coords"]
            #print("traces_alllalalalal",traces_all)
            #traces_all.sort(key=lambda trace_dict: int(trace_dict['id']))
            #traces_all = OrderedDict(sorted(traces_all.items(), key=lambda t: t['id']))
         
            return traces_all
        else:
            print('File ', inkml_file_abs_path, ' does not exist !')
            return {}
    
    
    #------------------------------------------------------------------------------#
    # an exemple of Output of train file iso4.inkml traces
    #[ 
    #    
    # [  [538, 187], [534, 189], [546, 189], [548, 189], [551, 188]],
    # [   [536, 178], [534, 178], [533, 178], [534, 178], [536, 178], [550, 179], [551, 179], [553, 181], [552, 182], [551, 182], [549, 183]]
    # 
    # ]
    #--------------------------------------------------------------------------------#

    def get_traces_data(self, traces_dict, id_set=None):
        'Accumulates traces_data of the inkml file'
        traces_data_curr_inkml = []
        if id_set == None:
            id_set = traces_dict.keys()
        #this range is specified by values specified in the lg file
        for i in id_set:  # use function for getting the exact range
            traces_data_curr_inkml.append(traces_dict[i])
            #print("trace for stroke"+str(i)+" :"+str(traces_data_curr_inkml))
            #image = self.convert_to_imgs(traces_data_curr_inkml, box_axis_size=box_axis_size)
        return traces_data_curr_inkml


    def convert_to_imgs(self, traces_data, box_axis_size):
        pattern_drawn = np.ones(
                shape=(box_axis_size, box_axis_size), dtype=np.float32)
        # Special case of inkml file with zero trace (empty)
        if len(traces_data) == 0:
            return np.matrix(pattern_drawn * 255, np.uint8)

        'mid coords needed to shift the pattern'
        #print("traces_all['coords']"+str(traces_data))
        min_x, min_y, max_x, max_y = self.get_min_coords(
        [item for sublist in traces_data for item in sublist])
    
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
        for traces_all in traces_data:
            'shift pattern to its relative position'
            shifted_trace = self.shift_trace(traces_all, min_x=min_x, min_y=min_y)
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
                centered_trace, pattern_drawn, box_axis_size=box_axis_size)

            #print("pattern_drawn :::::::::::\n{}".format(pattern_drawn))
            #print("pattern size", pattern_drawn.shape)
            #print(np.matrix(pattern_drawn, np.uint8))
        return np.matrix(pattern_drawn * 255, np.uint8)

    def get_min_coords(self, traces):
        x_coords = [coord[0] for coord in traces]
        #print("xcoords"+str(x_coords))
        y_coords = [coord[1] for coord in traces]
        min_x_coord = min(x_coords)
        min_y_coord = min(y_coords)
        max_x_coord = max(x_coords)
        max_y_coord = max(y_coords)
        return min_x_coord, min_y_coord, max_x_coord, max_y_coord


    'shift pattern to its relative position'

    def shift_trace(self, traces, min_x, min_y):
        shifted_trace = [[coord[0] - min_x, coord[1] - min_y] for coord in traces]
        return shifted_trace


    'Scaling: Interpolates a pattern so that it fits into a box with specified size'


    def scaling(self, traces, scale_factor=1.0):
        interpolated_trace = []
        'coordinate convertion to int type necessary'
        interpolated_trace = [
            [round(coord[0] * scale_factor), round(coord[1] * scale_factor)] for coord in traces]
        return interpolated_trace


    def center_pattern(self, traces, max_x, max_y, box_axis_size):
        x_margin = int((box_axis_size - max_x) / 2)
        y_margin = int((box_axis_size - max_y) / 2)
        return self.shift_trace(traces, min_x=-x_margin, min_y=-y_margin)

    def draw_pattern(self, traces, pattern_drawn, box_axis_size):
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

    
    #hepl to sort glob glob 
    def sortGlobFilesArray(self, s):
        return int(os.path.basename(s)[0:-6])


if __name__ == '__main__':
    train_img_instance = InkDataExtractor()
