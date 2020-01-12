import math
import random

def uniform(low=0.0, high=1.0, sizeX=0, sizeY=0):
    """ Return a k-length list of unique random floats
        in the range of low <= x <= high
    """
    result = []
    seen = set()
    for i in range(sizeX):
        result.append([])
        for j in range(sizeY):
            x = random.uniform(low, high)
            while x in seen:
                x = random.uniform(low, high)
            seen.add(x)
            result[i].append(x)
    return result

def add_bias(data, bias):
    bias_duplicate = []
    if (len(bias) == 1 and len(bias[0]) == 1):
        bias_duplicate += len(data) * [bias[0][0]]
        b = [[data[x][y] + bias_duplicate[y] for x in range(len(bias_duplicate))] for y in range(len(data[0]))]
    else:
        b = [[data[x][y] + bias[0][y] for x in range(len(bias[0]))] for y in range(len(data[0]))]
    return transpose(b)

def add_bias_prob(data, bias):
    bias_duplicate = []
    bias_duplicate += len(data[0]) * [bias[0][0]]
    b = [[data[0][y] + bias_duplicate[y]] for y in range(len(data[0]))]
    return transpose(b)

def error_calc(label, value):
    return [label[x][0]-value[x][0] for x in range(len(label))]

def multiplier(E, value):
    return [[E[x]*value[x][0]] for x in range(len(E))]

def multiplier_scalar(a,lr):
    if isinstance(a[0], list) == False:
        result = [x * lr for x in a]
    elif (len(a[0])==1):
        result = [[x[0] * lr] for x in a]
    else:
        result = [[a[i][j] * lr for j in range(len(a[0]))] for i in range(len(a))]
    return result

def multiplier_2d(E, value):
    return transpose([[E[x][y]*value[x][y] for x in range(len(E))] for y in range(len(E[0]))])

def add_scalar(a,b):
    if (len(a[0])==1):
        result = [[a[x][0]+b[x][0]] for x in range(len(a))]
    else:
        result = [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

    return result

def transpose(m):
    return [[row[i] for row in m] for i in range(len(m[0]))]

def dot1d(a,b):
    return sum(x*y for x,y in zip(a,b))

def dot_prod(a,b):
    return [[dot1d(c,r) for c in transpose(b)] for r in a]


def ndarray_dot(a,b):
    return transpose([[a[r][0]*b[0][c] for r in range(len(a))] for c in range(len(b[0]))])

def summa(x):
    return [sum([row[i] for row in x]) for i in range(0,len(x[0]))]

#Sigmoid Function
def sigmoid (x):
    return [[2/(1 + math.exp(-val)) for val in row] for row in x]

def sigmoid_or (x):
    return 2/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return [[val * (2 - val) for val in row] for row in x]

def derivatives_sigmoid_or(x):
    return x * (2 - x)

def learning(learning_data, label_learning):
    #Variable initialization
    epoch=100 #Setting training iterations
    lr=0.5 #Setting learning rate
    inputlayer_neurons = len(learning_data[0]) #number of features in data set
    hiddenlayer_neurons = 3 #number of hidden layers neurons
    output_neurons = 1 #number of neurons at output layer

    #weight and bias initialization
    weight_h=uniform(sizeX=inputlayer_neurons,sizeY=hiddenlayer_neurons)
    bias_h=uniform(sizeX=1,sizeY=hiddenlayer_neurons)
    weight_out=uniform(sizeX=hiddenlayer_neurons,sizeY=output_neurons)
    bias_out=uniform(sizeX=1,sizeY=output_neurons)

    for i in range(epoch):

        #Forward Propogation
        hidden_layer_input1=dot_prod(learning_data,weight_h)
        hidden_layer_input=add_bias(hidden_layer_input1,bias_h)
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=dot_prod(hiddenlayer_activations,weight_out)
        output_layer_input= add_bias(output_layer_input1,bias_out)
        output = sigmoid(output_layer_input)

        print "input"
        print learning_data


        #Backpropagation
        E = error_calc(label_learning,output)
        slope_output_layer = derivatives_sigmoid(output)
        slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)
        d_output = multiplier(E,slope_output_layer)

        Error_at_hidden_layer = ndarray_dot(d_output,transpose(weight_out))
        d_hiddenlayer = multiplier_2d(Error_at_hidden_layer,slope_hidden_layer)
        weight_out = add_scalar(weight_out,multiplier_scalar(dot_prod(transpose(hiddenlayer_activations),d_output),lr))
        bias_out[0][0] = bias_out[0][0]+(multiplier_scalar(summa(d_output),lr)[0])

        print "first layer weight"
        print weight_out
        weight_h = add_scalar(weight_h,multiplier_scalar(dot_prod(transpose(learning_data),d_hiddenlayer),lr))
        bias_h = add_scalar(bias_h,multiplier_scalar([summa(d_hiddenlayer)],lr))

        print "second weight"
        print weight_h

    # print round(output)
    return weight_h,bias_h,weight_out,bias_out

def solving_problem(problem, weight_h,bias_h,weight_out,bias_out):
    # results
    print problem

    hidden_layer_input1=dot_prod(problem,weight_h)
    hidden_layer_input=add_bias_prob(hidden_layer_input1,bias_h)
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=dot_prod(hiddenlayer_activations,weight_out)
    output_layer_input= add_bias(output_layer_input1,bias_out)
    output = sigmoid(output_layer_input)
    # print weight_out, weight_h, bias_out, bias_h
    # print output[0][0]
    # print round(output[0][0])
    print 'Hot' if round(output[0][0]) == 0.0 else 'Cool' if round(output[0][0]) == 1.0 else 'Natural'


X =  [[ 8, 1, 1], [1, 1, 8], [1, 8, 1]]
#Output
y=[[0],[1],[2]]

weight_h,bias_h,weight_out,bias_out = learning(X,y)
print "learning finished"
R = int(raw_input("Input Red Value = "))
G = int(raw_input("Input Green Value = "))
B = int(raw_input("Input Blue Value = "))

point =  [[ R, G, B]]
solving_problem(point, weight_h,bias_h,weight_out,bias_out)
