from oneHiddenNet import One_Hidden_Layer_Network

Network = One_Hidden_Layer_Network()
#Check the lower level functions and try sanity checks (for debuggin purposes)
Network.check()
#Train a Simple Logistic Regression model to it fails on non-linear data
Network.simpleLogReg()
#Execute and train the network with one hidden layer on actual dataset and test on the test set
Network.execute()
#Try different number of nodes in hidden layer to find the best network structure
Network.opt_hidden()