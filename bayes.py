import sys
import math
import arff
import random
import matplotlib.pyplot as plt

def parse_file(filename):
    trainingData = arff.load(open(filename,'rb'))
    trainingInstances = trainingData['data']
    attributes = trainingData['attributes']
    return attributes,trainingInstances


def main():
    argument_list=(sys.argv)
    trainfile_name = argument_list[1]
    testfile_name = argument_list[2]
    attributes,training_set = parse_file(trainfile_name)
    test_attribute,test_set = parse_file(testfile_name)
    n_or_t = argument_list[3]
    if n_or_t == 'n':
        naive_bayes(attributes,training_set,test_set)
    elif n_or_t == 't':
        TAN_bayes(attributes,training_set,test_set)
    else:
        print 'Enter either n or t'
    #-------------------Plotting--------------------------------
    #plotForSubset(attributes,training_set,test_set)

def TAN_bayes(attributes,training_set,test_set):
    count_list,class_count = calculate_count(attributes,training_set)
    class1_prob = float(class_count[0]+1)/float(len(training_set)+2)
    class2_prob = float(class_count[1]+1)/float(len(training_set)+2)
    correct_count = 0
    MST = find_MST_prims(attributes,training_set)
    joint_prob_list = calculate_joint_laplace_prob(MST,training_set,attributes)
    print attributes[0][0],attributes[-1][0]
    for i,features in enumerate(attributes[1:-1]):
        parent_feature = [item[0] for item in MST[1:] if item[1]==i+1]
        print features[0],attributes[parent_feature[0]][0],attributes[-1][0]
    for data in test_set:
        class1_prob_product = joint_prob_list[0][data[0]][0]
        class2_prob_product = joint_prob_list[0][data[0]][1]
        feature_vector = []
        for feature in data[:-1]:
            feature_vector.append(feature)
        for i,feature in enumerate(data[1:-1]):
            feature_2_index = [item[0] for item in MST[1:] if item[1]==i+1]
            parent_feature = data[feature_2_index[0]]
            class1_prob_product = float(joint_prob_list[i+1][feature][parent_feature][0])*class1_prob_product
            class2_prob_product = float(joint_prob_list[i+1][feature][parent_feature][1])*class2_prob_product
        numerator = float(class1_prob)*float(class1_prob_product)
        denominator = numerator + float(class2_prob)*float(class2_prob_product)
        conditional_prob = float(numerator)/float(denominator)
        if conditional_prob >= 0.5:
            print attributes[-1][1][0], data[-1], conditional_prob
            if attributes[-1][1][0]==data[-1]:
                correct_count = correct_count+1
        else:
            print attributes[-1][1][1], data[-1], float(1 - conditional_prob)
            if attributes[-1][1][1] == data[-1]:
                correct_count = correct_count+1
    print '\n', correct_count
    return correct_count

#Takes the test data, calculates conditional probability based on probabilitiies calculated using traning set and then classify it
#based on the value of bayes formula
def naive_bayes(attributes,training_set,test_set):
    count_list,class_count = calculate_count(attributes,training_set)
    class1_prob = float(class_count[0]+1)/float(len(training_set)+2)
    class2_prob = float(class_count[1]+1)/float(len(training_set)+2)
    prob_list = calculate_laplace_prob(count_list,class_count)
    correct_count = 0
    for i,attribute in enumerate(attributes[:-1]):
         print attribute[0],'class'
    for data in test_set:
        class1_prob_product = 1
        class2_prob_product = 1
        for i,feature in enumerate(data[:-1]):
            class1_prob_product = prob_list[i][feature][0]*class1_prob_product
            class2_prob_product = prob_list[i][feature][1]*class2_prob_product
        numerator = class1_prob*class1_prob_product
        denominator = numerator + class2_prob*class2_prob_product
        conditional_prob = float(numerator)/float(denominator)
        #print "conditional_prob", conditional_prob
        if conditional_prob >= 0.5:
            print attributes[-1][1][0], data[-1], conditional_prob
            if attributes[-1][1][0]==data[-1]:
                correct_count = correct_count+1
        else:
            print attributes[-1][1][1], data[-1], float(1 - conditional_prob)
            if attributes[-1][1][1] == data[-1]:
                correct_count = correct_count+1
    print '\n', correct_count
    return correct_count

#For tan bayes
def calculate_joint_laplace_prob(MST,training_set,attributes):
    count_list,class_count = calculate_count(attributes,training_set)
    prob_list = calculate_laplace_prob(count_list,class_count)
    joint_prob_list = []
    for feature in attributes[:-1]:
        joint_prob_list.append({})
    for element in attributes[0][1]:
        joint_prob_list[0].update({element:prob_list[0][element]})
    for edge in MST[1:]:
        joint_count_list = calculate_joint_counts(attributes[edge[1]],attributes[edge[0]],training_set,attributes)
        all_xi_values = len(joint_count_list)
        j = attributes.index(attributes[edge[1]])
        for i,element in enumerate(attributes[edge[1]][1]):
            joint_prob_list[j].update({element:{}})
            for element1 in attributes[edge[0]][1]:
                numerator1 = joint_count_list[i][element1][0]+1
                numerator2 = joint_count_list[i][element1][1]+1
                denominator1 = count_list[attributes.index(attributes[edge[0]])][element1][0] + all_xi_values
                denominator2 = count_list[attributes.index(attributes[edge[0]])][element1][1] + all_xi_values
                class1_prob = float(numerator1)/float(denominator1)
                class2_prob = float(numerator2)/float(denominator2)
                class_prob = [class1_prob,class2_prob]
                joint_prob_list[j][element].update({element1:class_prob})
    return joint_prob_list
#For naive bayes
def calculate_laplace_prob(count_list,class_count):
    prob_list = []
    for i,features in enumerate(count_list):
        prob_list.append({})
        for elements in features:
            numerator1 = features[elements][0]+1
            denominator1 = class_count[0]+len(features)
            numerator2 = features[elements][1]+1
            denominator2 = class_count[1]+len(features)
            class1_laplace_cond_prob = float(numerator1)/float(denominator1)
            class2_laplace_cond_prob = float(numerator2)/float(denominator2)
            prob_list[i].update({elements:[class1_laplace_cond_prob,class2_laplace_cond_prob]})
    return prob_list


#Finds the MST using prim's for a set of nodes where each node is a feature
def find_MST_prims(attributes,training_set):
    mutual_information_list = calculate_mutual_information_list(attributes,training_set)
    visited_nodes = []
    initial_nodes = []
    max_edge_weight = -1
    max_edge_index = -1
    source_node = -1
    MST = [()]
    for i,attribute in enumerate(attributes[:-1]):
        initial_nodes.append(i)
    initial_nodes_duplicate = initial_nodes[:]
    visited_nodes.append(initial_nodes_duplicate[0])
    initial_nodes.remove(0)
    while len(initial_nodes)>0:
        max_edge_weight = -1
        for node in visited_nodes:
            for weight in mutual_information_list[node]:
                index = mutual_information_list[node].index(weight)
                if index not in visited_nodes and weight>max_edge_weight:
                    max_edge_weight = weight
                    max_edge_index = index
                    source_node = node
        visited_nodes.append(initial_nodes_duplicate[max_edge_index])
        initial_nodes.remove(max_edge_index)
        MST.append((source_node,visited_nodes[-1]))
    return MST

#Calculated the matrix of mutual information between various features
def calculate_mutual_information_list(attributes,training_set):
    count_list,class_count = calculate_count(attributes,training_set)
    class1_prob = float(class_count[0]+1)/float(len(training_set)+2)
    class2_prob = float(class_count[1]+1)/float(len(training_set)+2)
    class_prob = [class1_prob,class2_prob]
    prob_list = calculate_laplace_prob(count_list,class_count)
    mutual_information_list = []
    for i,feature in enumerate(attributes[:-1]):
        mutual_information_list.append([])
        for j,other_feature in enumerate(attributes[:-1]):
            if i is not j:
                co_occurence_count = calculate_joint_counts(feature,other_feature,training_set,attributes)
                xi_sum = 0
                for k,element1 in enumerate(feature[1]):
                    xj_sum = 0
                    for element2 in other_feature[1]:
                        class_sum = 0
                        for l,classes in enumerate(class_prob):
                            numerator = float(co_occurence_count[k][element2][l] + 1)/float(class_count[l]+ len(feature[1])*len(other_feature[1]))
                            denominator = float(prob_list[i][element1][l])*float(prob_list[j][element2][l])
                            x = numerator/denominator
                            log_factor = math.log(x,2)
                            joint_prob = float(co_occurence_count[k][element2][l] + 1)/float(len(training_set)+
                                                                                             len(feature[1])*len(other_feature[1])*len(class_prob))
                            class_sum = class_sum + joint_prob*log_factor
                        xj_sum = xj_sum + class_sum
                    xi_sum = xi_sum+xj_sum
                mutual_information_list[i].append(xi_sum)
            else:
                mutual_information_list[i].append(-1)
    return mutual_information_list

#n(xi,xj,y)
def calculate_joint_counts(feature1,feature2,training_set,attributes):
    classes = attributes[-1]
    index1 = attributes.index(feature1)
    index2 = attributes.index(feature2)
    joint_counts_list = []
    for i,element1 in enumerate(feature1[1]):
        joint_counts_list.append({})
        for element2 in feature2[1]:
            joint_counts_list[i].update({element2:[0,0]})
            for data in training_set:
                if data[index1] == element1 and data[index2] == element2 and data[-1]==classes[1][0]:
                    joint_counts_list[i][element2][0] = joint_counts_list[i][element2][0]+1
                elif data[index1] == element1 and data[index2] == element2 and data[-1]==classes[1][1]:
                    joint_counts_list[i][element2][1] = joint_counts_list[i][element2][1]+1

    return joint_counts_list

#n(xi,y)
def calculate_count(attributes,training_set):
    classes = attributes[-1]
    class_count = [0,0]
    count_list = []
    for i,attribute in enumerate(attributes[:-1]):
        count_list.append({})
        for element in attribute[1]:
            count_list[i].update({element:[0,0]})
    for instance in training_set:
        if instance[-1] == classes[1][0]:
            class_count[0] = class_count[0]+1
            for j,feature in enumerate(instance[:-1]):
                count_list[j][feature][0] = count_list[j][feature][0]+1
        else:
            class_count[1] = class_count[1]+1
            for j,feature in enumerate(instance[:-1]):
                count_list[j][feature][1] = count_list[j][feature][1]+1
    return count_list,class_count

#---------------------------------------------Plotting Code ------------------------------------------------------------------------
def get_accuracy(attributes,training_data,test_data,size):
    accuracy_list = []
    new_training_data = []
    n_sum = 0
    t_sum = 0
    if size < len(training_data):
        for i in range(1,5):
            rand_smpl_indices = random.sample(xrange(len(training_data)), int(size))
            for index in rand_smpl_indices:
                new_training_data.append(training_data[index])
            n_accuracy = (naive_bayes(attributes,new_training_data,test_data))/float(len(test_data))
            t_accuracy = (TAN_bayes(attributes,new_training_data,test_data))/float(len(test_data))
            n_sum = n_sum+n_accuracy
            t_sum = t_sum+t_accuracy
        return [float(n_sum)/4.0,(t_sum)/4.0]

    else:
        rand_smpl_indices = random.sample(xrange(len(training_data)), int(size))
        for index in rand_smpl_indices:
            new_training_data.append(training_data[index])
        n_accuracy = naive_bayes(attributes,new_training_data,test_data)/float(len(test_data))
        t_accuracy = TAN_bayes(attributes,new_training_data,test_data)/float(len(test_data))
        return [n_accuracy,t_accuracy]

def plotForSubset(attributes,training_set,test_set):
    xvalues=[]
    nyvalues=[]
    tyvalues=[]
    for size in [25,50,100]:
        accuracies = get_accuracy(attributes,training_set,test_set,size)
        xvalues.append(size)
        nyvalues.append(accuracies[0])
        tyvalues.append(accuracies[1])
    plt.plot(xvalues,nyvalues,'r')
    plt.plot(xvalues,nyvalues,'ro')
    plt.plot(xvalues,tyvalues,'b')
    plt.plot(xvalues,tyvalues,'bo')
    plt.axis([20, 100, 0, 1])
    plt.xlabel('Data Size')
    plt.ylabel('Accuracy')
    plt.show()


