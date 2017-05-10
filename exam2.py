import DT
import treePloter
import numpy as np

fr = open('lenses.txt')
dataSet = [line.strip().split('\t')for line in fr.readlines()]
dataSet = np.array(dataSet)
attribute_list = ['age','prescript','astimatic','tearRate']
fixed_attribute_list = ['age','prescript','astimatic','tearRate']
tree = DT.createTree(np.array(dataSet),attribute_list,fixed_attribute_list)
print(tree)
#print(DT.classify(tree,['pre','myope','yes','normal','hard'],fixed_attribute_list))
#treePloter.createPlot(tree)