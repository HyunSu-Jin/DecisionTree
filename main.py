import sys

fr1 = open(sys.argv[1])
attribute_list = fr1.readline().strip().split('\t')
allColumn = attribute_list.copy()

attribute_list.pop(-1)

dataSet = [line.strip().split('\t') for line in fr1.readlines()]
#dataSet = np.array(dataSet)

dic_dataSet = [] # (dict,label)
for data in dataSet:
    # data is list
    label_name = attribute_list.pop(-1)
    label_val = data.pop(-1)
    dict_val = {}
    for feature,attribute in zip(data,attribute_list):
        dict_val[attribute] = feature
    dic_dataSet.append(dict_val,label_val)
