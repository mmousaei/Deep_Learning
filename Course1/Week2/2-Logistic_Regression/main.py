from LogReg import Logistic_Regression

Network = Logistic_Regression()
#Check the lower level functions and try sanity checks (for debuggin purposes)
Network.check(25)
#Execute and train the network on actual dataset and test on the test set
number_of_iterations = 200
learning_rate = 0.005
print_flag = True
Network.execute(number_of_iterations, learning_rate, print_flag)