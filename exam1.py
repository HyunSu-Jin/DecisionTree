import DT
import numpy as np

dataSet,labels = DT.createDataSet()
attribute_list = ['surface','flipper']
fixed_attribute = ['surface','flipper']
#mapToStr = ['Does it live on surface?','Does it have the flipper?']
tree =DT.createTree(np.array(dataSet),attribute_list,fixed_attribute)
#treePloter.createPlot(tree)
#print (DT.classify(tree,['1','1'],fixed_attribute))
print(tree)