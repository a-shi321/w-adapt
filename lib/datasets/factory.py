# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.water import water
import numpy as np
from datasets.vg import vg



for year in ['2007', '2012']:
  for split in ['train_trainval', 'train_test']:
    name = 'VOC_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year: pascal_voc(split, year))



for year in ['2007', '2012']:
  for split in ['train', 'val', 'train_all', 'test']:
    name = 'watercolor_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year:water(split, year))

def get_imdb(name):
  """Get an imdb (image database) by name."""
  #print("这个命名集合是个啥factory",__sets)
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
