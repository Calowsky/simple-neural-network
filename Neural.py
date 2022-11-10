#Neural Network Caleb Pekowsky 5/5/21

import numpy as np

# This is my ANN class. It has 4 input neurons, 4 hidden neurons, and 3 output neurons.
# I chose 4 input neurons and hidden neurons to represent the 4 possible inputs.
# I chose 3 output Neurons to represent the 3 types of iris flowers.
class ANN:
  def __init__(self):
    # input weights: 4x4 matrix 
    self.inputHiddenWeights = np.random.rand(4,4)
    # input biases: 4x1 matrix 
    self.inputHiddenBias = np.random.rand(4)
    # hidden layer: 3x4 matrix
    self.hiddenOutputWeights = np.random.rand(3,4)
    # hidden layer biases: 3x1 matrix 
    self.outputHiddenBias = np.random.rand(3)
    #these are really arbitrary values that can be tweaked by the user.
    self.learningRate = 0.10   
    self.maxIterations = 1000
    self.maxOverTrain = 10
    self.goalMSE = 0.01


#calls relevent functions
def main():

    #get data, which is the iris data set with the name "ANN - Iris data.txt"
    data = getData()
    #splits given data into training, testing, and validation
    training, testing, validation = splitData(data)
    #creates NN
    NN = ANN()
    #trains NN on the data
    finalNN = trainNeuralNetwork(training, validation, NN)
    #after training, we test the neural network.
    testNeuralNetwork(finalNN, testing)
    #allow user to input data, and let the NN classify it
    classifyInput(finalNN)


#tests the trained neural network on the testing data.
def testNeuralNetwork(NN, testing):
    accurate = 0
    testingMSE = 0
    for dataPoint in testing:
        realClassification, atributes = getrealClassificationAtributes(dataPoint)

        hiddenNeuronsOutput, finalNeuronsOutput = getOutputValues(NN, atributes)
        testingMSE += np.sum((realClassification - finalNeuronsOutput)**2)

        if realClassification[np.argmax(finalNeuronsOutput)] == 1:
            accurate += 1

    testingMSE /= 30
    print("Testing Accuracy: ", accurate / 30 * 100, "%")


#This actually trains the neural network 
def trainNeuralNetwork(training, validation, NN):

    #vals being checked and updated over the course of training
    numIterations = 0
    numOverTrain = 0
    previousValidationMSE = 1

    trainingMSE = 0
    validationMSE = 1
    
    # Training is run until:
    # 1: the goal Mean Square Error is reached
    # 2: the maximum iterations set by user is reached
    # 3: the data is over trained
    while validationMSE > NN.goalMSE and numIterations < NN.maxIterations and numOverTrain < NN.maxOverTrain:

        for dataPoint in training:
            
            #break the datapoint into atributes, and it's actual classification
            realClassification, atributes = getrealClassificationAtributes(dataPoint)
            
            #run forward propogation to get the NN's output on a datapoint, and update trainingMSE
            hiddenNeuronsOutput, finalNeuronsOutput, trainingMSE = forwardPropogation(NN, atributes, realClassification, trainingMSE)

            #run back propogation to update NN weights and biases
            backPropogation(finalNeuronsOutput, realClassification, atributes, NN, hiddenNeuronsOutput )

        #calculate the trainingMSE, useful to tell if data is overtrained
        trainingMSE /= 90

        #call the validation function
        validationMSE, numOverTrain = Validate(NN, validation, numIterations, numOverTrain, previousValidationMSE)
        
        previousValidationMSE = validationMSE
        numIterations+=1


    return NN  

#forward propogation: test an example on the NN, and return results
def forwardPropogation(NN, atributes, realClassification, trainingMSE):
    # apply an example to the input neurons, and calculate outputs
    hiddenNeuronsOutput, finalNeuronsOutput = getOutputValues(NN, atributes)
    #getting the squared error value
    trainingMSE += np.sum((realClassification - finalNeuronsOutput)**2)

    return hiddenNeuronsOutput, finalNeuronsOutput, trainingMSE


#updates NN weights and biases
def backPropogation(finalNeuronsOutput, realClassification, atributes, NN, hiddenNeuronsOutput ):
    # calculate the error in the final outputted array, as compared to the real classification
    finalError = activationDerivitive(finalNeuronsOutput) * (realClassification - finalNeuronsOutput)

    # recompute the weights between the hidden and output weights
    # NOTE: this linear algebra was complicated! I had to copy this logic from the internet.
    NN.hiddenOutputWeights += NN.learningRate * np.dot(finalError.reshape(3,1), hiddenNeuronsOutput.reshape(4,1).T)
    #recompute the bias using the finalError
    NN.outputHiddenBias += NN.learningRate * finalError

    # calculate the outputted error of the hidden layer
    # NOTE: this linear algebra was complicated! I had to copy this logic from the internet.
    hiddenOutputError = activationDerivitive(hiddenNeuronsOutput) * np.dot(NN.hiddenOutputWeights.T, finalError)

    # recompute the weights between the input and hidden weights
    # NOTE: this linear algebra was complicated! I had to copy this logic from the internet.
    NN.inputHiddenWeights += NN.learningRate * np.dot(hiddenOutputError.reshape(4,1), atributes.reshape(4,1).T)
    #recompute the bias using the finalError
    NN.inputHiddenBias += NN.learningRate * hiddenOutputError



# this function validates the neural network on a validation sample data set.
# It outputs the current validation accuracy, and keeps track of overtraining
def Validate(NN, validation,numIterations, numOverTrain, previousValidationMSE):
        validationMSE = 0

        # get the MSE of the validation data
        accurate = 0
        for dataPoint in validation:

            realClassification, atributes = getrealClassificationAtributes(dataPoint)

            hiddenNeuronsOutput, finalNeuronsOutput = getOutputValues(NN, atributes)
            validationMSE += np.sum((realClassification - finalNeuronsOutput)**2)

            if realClassification[np.argmax(finalNeuronsOutput)] == 1:
                accurate += 1

            
        validationMSE /= 30

        #this tests for over training by counting the number of times 
        # the validation error increases, but the testing error is close to what it should be 
        if validationMSE >= previousValidationMSE:
            numOverTrain += 1

        print("Validation Accuracy at iteration ", numIterations, ": " , accurate / 30 * 100, "%")
        return validationMSE, numOverTrain


#given potential, calculates derivitive of activation.
#useful for gradiant descent
def activationDerivitive(potential):
    return (activation(potential) * (1 - activation(potential)))

#given potential, calculates output
def activation(potential):
    return (1/(1 + (np.e ** (-potential))))

#splits data into training, test, and validation
def splitData(data):
    training = []
    testing = []
    validation = []

    training.extend(data[0:30])
    testing.extend(data[30:40])
    validation.extend(data[40:50])

    training.extend(data[50:80])
    testing.extend(data[80:90])
    validation.extend(data[90:100])

    training.extend(data[100:130])
    testing.extend(data[130:140])
    validation.extend(data[140:150])

    return training, testing, validation


#opens a file, and extracts the data.
def getData():
    data = []

    given = open("ANN - Iris data.txt", "r")
    data = given.read()
    given.close()

    data_into_list = data.split('\n')

    betterdata = []
    for currdata in data_into_list:
        if len(currdata) < 2:
            continue
        newdata = currdata.split(',')
        currbetterdata = []
        for i in range(0,4):
            currbetterdata.append(float(newdata[i]))

        if newdata[4] == 'Iris-setosa':
            currbetterdata.append(0)
        if newdata[4] == 'Iris-versicolor':
            currbetterdata.append(1)
        if newdata[4] == 'Iris-virginica':
            currbetterdata.append(2)

        betterdata.append(currbetterdata)
 
    #normalize to get all values between 0 and 1
    #assuming no values will ever be greater than 10
    finallarr = np.array(betterdata) / 10
    for arr in finallarr:
        arr[4] *= 10
    print(finallarr)

    return finallarr

#takes in user inputs, and uses NN to classify it.
def classifyInput(NN):
    while input != "quit":

        print("Neural Network has been trained. enter classification data. ")
        print("enter 'quit' to quit the program. ")
        print("Note: entering non-numbers or massive numbers breaks program ")
        while 1:
            Sepal_length = input("enter sepal length ")
            if(Sepal_length == "quit"):
                return
            Sepal_width = input("enter sepal width ")
            if(Sepal_width == "quit"):
                return

            Petal_length = input("enter petal length ")
            if(Petal_length == "quit"):
                return

            Petal_width = input("enter petal length ")
            if(Petal_width == "quit"):
                return

            Petal_length = float(Petal_length)
            Petal_width = float(Petal_width)

            Sepal_length = float(Sepal_length)
            Sepal_width = float(Sepal_width)

            UserArr = [ Sepal_length, Sepal_width, Petal_length, Petal_width]
            UserArr = np.asarray(UserArr) / 10

            hiddenNeuronsOutput, finalNeuronsOutput = getOutputValues(NN, UserArr)

            maxIndex = np.argmax(finalNeuronsOutput)
            if maxIndex == 0:   
                answer = "Iris-setosa"
            elif maxIndex == 1:   
                answer = "Iris-versicolor"
            elif maxIndex == 2:   
                answer = "Iris-virginica"

            print("Neural Network Prediction:", answer, "at", (np.max(finalNeuronsOutput)) * 100, "% ")

            

#calculates the output values of the NN, given an input
def getOutputValues(NN, currExample):
    hiddenPotentials = np.dot(NN.inputHiddenWeights, currExample) + NN.inputHiddenBias 

    hiddenNeuronsOutput = activation(hiddenPotentials)

    # potentials of output layer, and final output
    finalPotentials = np.dot(NN.hiddenOutputWeights, hiddenNeuronsOutput) + NN.outputHiddenBias
    finalNeuronsOutput = activation(finalPotentials)

    return hiddenNeuronsOutput, finalNeuronsOutput

# breaks data into an array representing the actual classification,
# and an array representing the 4 atributes.s
def getrealClassificationAtributes(dataPoint):
    atributes = dataPoint[0:4]

    #turning realClassification from int value to vector
    realClassification = [0]*3
    realClassification[ int(dataPoint[4]) ] +=1

    return realClassification, atributes


if __name__ == "__main__":
    main()
