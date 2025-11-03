from scipy.io import loadmat
import numpy as np
from dataset.utils.dataset_split import train_test_split

def load_tolokers():
    print("Loading tolokers data...")
    data = loadmat('tolokers.mat')
    print(data)

    A = data["Network"]  
    X = data["Attributes"]                         
    # y_class = data["Class"].flatten()              
    y_label = data["Label"].flatten()              
    # y_str = data["str_anomaly_label"].flatten()    # Redundant info in dataset, same as attribute anomaly (attr_anomaly_label)
    # y_attr = data["attr_anomaly_label"].flatten().astype(int)   

    print("Feature shape: ", X.shape)
    # print("Label shape: ", y_class.shape)  # using Class
    # print("Unique Labels (Class): ", np.unique(y_class))
    print("Unique Labels (Label): ", np.unique(y_label))
    # print("Unique labels in structural anomaly: ", np.unique(y_str))
    # print("count of structural anomaly: ", int(y_str.sum()))
    # print("Unique labels in attribute anomaly: ", np.unique(y_attr))
    # print("count of attribute anomaly: ", int(y_attr.sum()))
    # print("Percentage of anomaly: ", round(int(y_attr.sum())/len(y_attr) * 100, 4))
    print("Percentage of anomaly: ", round(int(y_label.sum())/len(y_label) * 100, 4))
    print("=============================================")

    X_train, train_edge_index, y_train, X_test, test_edge_index, y_test=train_test_split(X,A,y_label)
    return X_train, train_edge_index, y_train, X_test, test_edge_index, y_test
