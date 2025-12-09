import pandas
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('nyc_bicycle_counts_2016.csv')
dataset_2['Brooklyn Bridge']      = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True))
dataset_2['Manhattan Bridge']     = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge']    = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge']  = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Total']                = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))
dataset_2['High Temp']            = pandas.to_numeric(dataset_2['High Temp'].replace(',','', regex=True))
dataset_2['Low Temp']             = pandas.to_numeric(dataset_2['Low Temp'].replace(',','', regex=True))
dataset_2['Precipitation']         = pandas.to_numeric(dataset_2['Precipitation'].replace(',','', regex=True))

x_samples = [dataset_2["Brooklyn Bridge"], dataset_2['Manhattan Bridge'], dataset_2['Queensboro Bridge'], dataset_2['Williamsburg Bridge']]
x_labels = ["Brooklyn Bridge", 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']



def getRelevant():
    global x_samples
    scores = []
    
    for i in range(4):
        x = np.column_stack(x_samples[1:])
        x_samples = x_samples[1:] + x_samples[:1]
        linear_model = LinearRegression().fit(x, dataset_2['Total'])
        scores.append(linear_model.score(x, dataset_2['Total']))

    print(scores)
    print(f"Sensor should not be on {x_labels[scores.index(min(scores))]}")

def weather (dataset_2):

    X = dataset_2[["High Temp", "Low Temp", "Precipitation"]]
    y = dataset_2["Total"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = linear_model.Ridge()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    train_preds = model.predict(X_train)
    print(f"Best alpha: {model.alpha}")
    r2 = r2_score(y_test, preds)
    #mse = mean_squared_error(y_test, preds)
    train_mse = mean_squared_error(y_train, train_preds)

    print("R²:", r2)
    print("Train MSE:", train_mse)
    print("Weather data is not predictive because even though R^2 value is larger than 0.5 (0.585>0.5), the model's MSE is 16700664 " \
    "which is really large even when Ridge is applied.")



getRelevant()
weather(dataset_2)


######################################
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Import scikit-learn metrics module for accuracy and AUROC calculation
from sklearn import metrics

def conf_matrix(y_pred, y_true, num_class):
    """
    agrs:
    y_pred : List of predicted classes
    y_true : List of corresponding true class labels
    num_class : The number of distinct classes being predicted

    Returns:
    M : Confusion matrix as a numpy array with dimensions (num_class, num_class)
    """
    # Your code here. We ask that you not use an external library like sklearn to create the confusion matrix and code this function manually

    # every row = actual value, each col = predicted value
    # reference code from stackoverflow https://stackoverflow.com/questions/61193476/constructing-a-confusion-matrix-from-data-without-sklearn
    
    #initializing confusion matrix with all zeros
    confuse = np.zeros((num_class, num_class))
    unique_class = np.unique (y_true)
    #loop accross the different combinations of actual/predicted class
    for i in range (num_class):
        for j in range (num_class):
            #count the number of instances in each combination of actual /predicted classes
            confuse [i,j] = np.sum((y_true==unique_class[i])&(y_pred ==unique_class[j]))
    return confuse 

def get_model(name, params):
    """
    args:
    name : Model name (string)
    params : list of parameters corresponding to given model

    Returns:
    model : sklearn model object
    """
    model = None
    if name == "KNN":
        k = params # Note that the expected parameters have already been extracted here
        # Define KNN model using sklearn KNeighborsClassifier object 
        # Note: you should include n_neighbors=k as an argument when initializing the model
        #reference code from scikitlearn
        model = KNeighborsClassifier (n_neighbors=k)
    elif name == "SVM":
        rand_state, prob = params # Note that the expected parameters have already been extracted here
        # Define SVM model using sklearn SVC object
        # Note: you should include random_state=rand_state and probability=prob as arguments when initializing the model
        model = SVC(random_state=rand_state,probability=prob)
    elif name == "MLP":
        hl_sizes, rand_state, act_func = params # Note that the expected parameters have already been extracted here
        # Define MLP model using sklearn MLPClassifier object
        # Note: you should include hidden_layer_sizes=hl_sizes, random_state=rand_state, and activation=act_func when initializing the model
        model = MLPClassifier(hidden_layer_sizes=hl_sizes, random_state=rand_state, activation=act_func)
    else:
        print("ERROR: Model name not recognized/supported. Returned None")

    return model

def get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class):
    """
    args:
    model_name : Model name as a string
    params : List of parameters corresponding to the given model 
    train_Data : 5000x784 numpy array of FMNIST training images
    train_labels : corresponding 5000 numpy array of strings containing ground truth labels
    test_Data : 1000x784 numpy array of FMNIST test images
    test_labels : corresponding 1000 numpy array of strings containing ground truth labels
    num_class : integer number of unique classes being predicted

    Returns: 
    accuracy : Total model accuracy (numpy float) 
    confusion matrix: numpy array of dimensions (num_class,num_class)
    auc_score : Area under the curve of the ROC metric (numpy float)
    """
    # 1. Create Classifier model
    model = get_model(model_name, params)

    # 2. Train the model using the training sets 
    model.fit(train_data, train_labels)
    # 3. Predict the response for test dataset
    predict = model.predict(test_data)
    # 4. Model Accuracy, how often is the classifier correct? You may use metrics.accuracy_score(...)
    acc = metrics.accuracy_score(y_true=test_labels,y_pred=predict)
    # 5. Calculate the confusion matrix by using the completed the function above 
    conf_mat = conf_matrix(predict,test_labels,num_class)
    # 6. Compute the AUROC score. You may use metrics.roc_auc_score(...)
    auc_scor = metrics.roc_auc_score(y_true=test_labels,y_score=model.predict_proba(test_data),multi_class='ovr')
    #acc, conf_mat, auc_scor = None, None, None # DELETE THIS LINE ONCE YOU HAVE CODED YOUR RESULTS 
    return acc, conf_mat, auc_scor


# if __name__ == "__main__":
#     train_data, train_labels, test_data, test_labels = load_Dataset()
#     num_class = 10
    
#     model_name = "KNN"
#     for k in range(1,6):
#         print(str(k)+"-neighbors result:")
#         params = k
#         accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class)
#         print("Accuracy:", accuracy)
#         print("AUROC Score:", auc_score)
#         print(confusion_matrix)
#         print()
        
#     model_name = "SVM"
#     params = [1, True]
#     accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class)
#     print("SVM Result")
#     print("Accuracy:", accuracy)
#     print("AUROC Score:", auc_score)
#     print(confusion_matrix)
#     print()
    
#     model_name = "MLP"
#     params = [(15,10), 1, "relu"]
#     accuracy, confusion_matrix, auc_score = get_model_results(model_name, params, train_data, train_labels, test_data, test_labels, num_class)
#     print("MLP Result")
#     print("Accuracy:", accuracy)
#     print("AUROC Score:", auc_score)
#     print(confusion_matrix)