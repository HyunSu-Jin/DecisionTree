# DecisionTree
decision tree classifier implemented by python3

## 정의
Decision-Tree는 머신러닝기법을 이용한 supervised learing의 한 종류이다. dataset으로부터 training dataset, test dataset으로 분류하여 모델을 학습한다.
DT는 tree 구조를 가져, input tuple이 각 노드에 해당하는 feature에 해당하는 값과 해당 노드가 가진 branches 와 일치하는 것을 따라가는 과정을 leaf-node에 도달할 때 까지 반복한다. 이러한 분류기법은 KNN, regression방법과는 다르게 별도의 background없이도 logic에 대한 이해가 쉽다는 장점을 가지고 있다. 또한, 구현과정이 비교적 간편함에도 불구하고 classify 성능이 work well 하다는 컴퓨터 과학자들의 평가가 있다.

## 주요사항
1. DT에서 중요한 사항은 트리의 노드의 level이 낮은것, 즉 트리의 꼭대기쪽에 위치하는 노드들이 얼마나 데이터를 효율적으로 나누냐가 성능의 관건이다. node의 level이 낮을수록 그 노드에서 나눈 partion은 DT의 전체적 성능에 큰 영향력을 미친다.

2. 따라서, "데이터를 가장 효율적으로 분류시키는 feature(attribute)를 선택"하는 것이 관건인데 이에 대한 해결책으로 ID3기법을 소개하고자 한다.

3. data의 각 feature들에 대해 feature가 nomial, binary, ordinal과 같은 categorical 형식으로 불연속적인 경우, 각각의 경우에 대해 branching 하며, numeric(continuous) 형식의 feature인 경우 spliting point를 정의하여, 해당 범위에 속하는 값에 대해 branching을 수행한다.

## Information Gain, Gain by ID3 method
Information Gain(정보 소득)은 해당 노드에 해당하는 Dataset partition(D)가 얼마나 순수한지를 나타낸다. '순수하다'라는 것의 척도는 class lable이 서로 다른 tuple이 얼마나 혼재되어있느냐를 나타낸다. D에 속한 모든 tuple이 class lable이 같다면 Information Gain,Info(D)는 0이고 모두 다르다면 1이다. 즉, Infomation Gain을 최소화 시키는 것이 Decision-tree의 각 branching과정의 목적이다.
불순도를 표현하기 위해서 아래와 같은 형태인 "Entropy"를 사용한다.
pi의 정의는 다음과 같다.
The probability that a tuple in D belongs Ci.
Ci means the one of class lable
![img](/img/note.jpg)
현재 노드의 Info(D)와 현재 노드에서 선택할 수 있는 모든 attribute에 대한
Information Gain ( InfoA(D) ) 를 구해 가장 최소의 InfoA(D)를 만족시키는 attribute(feature)를 선택한다.
위 사진에서 최소의 InfoA(D)를 선택하는 것은 최대의Gain(A)를 선택하는 것과 완전히 동일한 뜻이다.

## Generate decision tree
트리를 만드는 과정은 앞서 설명한 방법을 recursive 하게 적용하는 것인데, recursive에 대한 기저사례는 다음과 같다.
1. 현재 D에 속한 모든 tuple이 같은 class lable을 갖는 경우
2. branching을 수행할 attribute가 없는 경우
3. 현재 D에 속한 tuple이 없는 경우

2상태에서는 해당 D에 속한 tuple이 pure하지 않지만 더이상의 branching이 불가능하므로 해당노드를 가장 숫자가 많은 class lable로 지정한다. ( 이 과정에서 Decision -tree의 accuracy가 저하된다.)

## 주요 소스코드
1. create Tree
<pre><code>
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
</code></pre>

2. Majority count
<pre><code>
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
</code></pre>

3. Information Gain
<pre><code>
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
</code></pre>

## 성능이슈
Decision tree는 위 예제와 같이 feature의 숫자가 적고, 각 feature들에 대해 branching되는 branching factor의 수가 적은경우 성능이 만족스럽게 나타난다. 그러나, branching factor가 대용량이고 feature의 숫자가 많은 경우 search space가 너무 커져 프로그램이 memory상에 올라갈 수 없게 될수도 있고, test-tuple에 대한 prediction이 너무 오래 걸릴 가능성이 있다.
이에 대한 해결책으로 tree를 pruning하여 search space를 줄인다거나, Attribute를 선택하는 방법으로 ID3대신 CA.5의 Gain Ratio 또는 CART의 Gini index를 사용한다.

그 외에, decision tree성능에 data의 성질이 영향을 주기도 하는데 만약 data가 가진 각 feature들이 서로 correlative되어 있다면 positive, negative한 경우는 상관없이 각 branching과정에 대한 entropy를 줄이는 데 실질적으로 연관된 속성중 한개만 영향력을 행사하는 꼴이므로 쓸데없이 searh space만 커지고 성능개선은 기대할 수 없다
또한, Decision-tree는 Machine learning의 고질적인 문제인 overfitting에도 취약한데, training dataset에 너무 specific하게 학습한 나머지 test dataset에 대해 낮은성능을 보이는 경우를 의미한다.

따라서, 모든 데이터마이닝 과정에 공통적으로 해당되는 말이겠으나, Decision-tree과정을 진행하기 전에 dataset에 대해 충분한 '데이터 전처리'를 수행하여 앞서 언급했던 correlative feature 또는 noise를 제거하여 classifier의 성능을 최대한 끌어올려야 한다.

## matplotlib를 이용한 tree ( lenses.txt 참조)
![tree](/img/tree.png)
