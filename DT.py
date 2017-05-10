from math import log
import numpy as np
import collections
import operator

def createDataSet():
    dataSet = [
        ['1','1','yes'],
        ['1','1','yes'],
        ['1','0','no'],
        ['0','1','no'],
        ['0','1','no']
    ]
    labels = ['no surfacing','flippers']
    return np.array(dataSet), np.array(labels)

def majorityCnt(dataSet):
    labels = np.unique(dataSet[:,-1])
    labelCnt = collections.defaultdict(int)
    for label in labels:
        bool_arr = dataSet[:,-1] == label
        matched = dataSet[bool_arr]
        matched_num = matched.shape[0]
        labelCnt[label] = matched_num
    labelCnt = sorted(labelCnt.items(),key=operator.itemgetter(1),reverse=True)
    return labelCnt[0][0]

def entropy(dataSet):
    m = dataSet.shape[0]
    labels = dataSet[:,-1]
    labels = np.unique(labels)
    sum = 0.0
    for label in labels:
        bool_arr = dataSet[:,-1] == label
        matched = dataSet[bool_arr]
        matched_num = matched.shape[0]
        prob = matched_num / m
        sum += prob * log(prob,2)
    return -sum



def deleteColumnIdx(rIdx,dataSet):
    return np.delete(dataSet,rIdx,axis=1)

def getAttributeIdx(feature,fixed_attribute):
    return fixed_attribute.index(feature)

def chooseBestFeatureToSplit(dataSet,attribute_list,fixed_attribute):
    features_num = len(attribute_list)
    m = dataSet.shape[0]
    entropies = collections.defaultdict(int)
    for i in range(features_num):
        sum = 0.0
        feature_name = attribute_list[i]
        idx = getAttributeIdx(feature_name,fixed_attribute)
        feature_vals = dataSet[:,idx]
        uniques = np.unique(feature_vals)
        for value in uniques:
            bool_arr = feature_vals == value
            matched = dataSet[bool_arr]
            matched_num = matched.shape[0]
            sum += (matched_num / m) * entropy(matched)
        entropies[feature_name] = sum
    entropies = sorted(entropies.items(),key=operator.itemgetter(1))
    attribute_list.remove(entropies[0][0])
    return entropies[0][0]

def createTree(dataSet,attribute_list,fixed_attribute):
    labels = dataSet[:,-1]
    if len(np.unique(labels)) == 1:
        return labels[0] # dataSet에 속한 모든 tuple이 같은 class label을 지닌 경우
    if len(attribute_list) == 0:
        return majorityCnt(dataSet)
    selected_feature = chooseBestFeatureToSplit(dataSet,attribute_list,fixed_attribute)
    idx = getAttributeIdx(selected_feature,fixed_attribute)
    msg = fixed_attribute[idx]
    mytree = { msg : {}}
    unique = np.unique(dataSet[:,idx])
    for feature_val in unique:
        bool_arr = dataSet[:,idx] == feature_val
        matched = dataSet[bool_arr]
        mytree[msg][feature_val] = createTree(matched,attribute_list,fixed_attribute)
    return mytree

def classify(tree,testVec,fixed_attribute):
    name = list(tree.keys())[0] # get name
    subtree = tree[name] # get the subtrees
    #print(subtree)
    idx = getAttributeIdx(name,fixed_attribute)
    for key in subtree.keys():
        if testVec[idx] == key: # sample에 해당 feature에 해당하는 branch를 찾았으면
            if type(subtree[key]).__name__ == 'dict':
                prediction = classify(subtree[key],testVec,fixed_attribute)
            else:
                prediction = subtree[key]
    return prediction
