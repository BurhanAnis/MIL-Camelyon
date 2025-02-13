import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models


class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', transform = None):
        lib = torch.load(libraryfile)
        slides = []
        for i, name in enumerate(lib['slides']):
            sys.stdout.write('Opening TIF headers: [{}/{}]\r'.format(i+1, len(lib['slides'])))
            sys.stdout.flush()
            slides.append(openslide.OpenSlide(name))
        print('')
        grid = []
        slideIDX = []
        for i, g in enumerate(lib['grid']):
            grid.extend(g)
            slideIDX.extend([i]*len(g)) #flatten grid into single list and keep track of IDX

        print('Number of tiles: {}'.format(len(grid)))

        self.slidenames = lib['slides']
        self.slides = slides
        self.targets = lib['targets']
        self.grid = grid
        self.slideIDX = slideIDX
        self.transform = transform
        self.mode = None
        self.mult = lib['mult']
        self.size = int(np.round(224*lib['mult']))
        self.level = lib['level']

    def setmode(self, mode):
        self.mode = mode

    def maketraindata(self, idxs):
        """make list of tuples with each tuple: (slide_index, tile_coordinates, slide_label)"""
        self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]

    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data)) #select from train data in random order


    def __getitem__(self,index):
        if self.mode == 1: #inference mode
            slideIDX = self.slideIDX[index]
            coord = self.grid[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2: #train, validation mode
            slideIDX, coord, target = self.t_data[index]
            img = self.slides[slideIDX].read_region(coord,self.level,(self.size,self.size)).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224),Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        if self.mode ==1:
            return len(self.grid)
        elif self.mode ==2:
            return len(self.t_data)

