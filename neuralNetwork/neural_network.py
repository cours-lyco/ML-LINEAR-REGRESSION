#!/usr/bin/env python3
import numpy as np 


class InkDataExtractor(object):
    path:
    
    def __init__(self):
        
    def readInkFile(ink_file_abs_path):
        with open(ink_file_abs_path, 'r') as fileObject:
            lines = fileObject.readlines()
            print(lines)

if __name__ == '__main__':
    inkdatas = InkDataExtractor()
