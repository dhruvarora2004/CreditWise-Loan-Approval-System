from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
def lr():
    #Pipelines 
    pipeline_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr",LogisticRegression())
    ])

    #parameters
    param_lr = {
        'lr__C': [0.01, 0.1, 1, 10,50,100]
    }

    #Classifier
    classifier_lr = GridSearchCV(pipeline_lr, param_lr, cv=5, scoring='precision')
    return classifier_lr
    


def nb():
    #Pipelines 
    pipeline_nb = Pipeline([
        ("scaler", StandardScaler()),
        ("nb",GaussianNB())
    ])

    #parameters
    param_nb={
        
    }

    #Classifier
    classifier_nb = GridSearchCV(pipeline_nb, param_nb, scoring='precision')
    return classifier_nb
def knn():  
    #Pipelines 
    pipeline_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("knn",KNeighborsClassifier())
    ])
    
    #parameters
    param_knn = {
        'knn__n_neighbors': [3, 5, 7, 9]
    }
    
    #Classifier
    classifier_knn = GridSearchCV(pipeline_knn, param_knn, cv=5, scoring='precision')
    return classifier_knn



def evaluate_model(model_name, y_test, y_pred):
    print(model_name)
    print(f"Accuracy Score {model_name}: {accuracy_score(y_test, y_pred)}")
    print(f"Precision Score {model_name}: {precision_score(y_test, y_pred)}")
    print(f"Recall Score {model_name}: {recall_score(y_test, y_pred)}")
    print(f"F1 Score {model_name}: {f1_score(y_test, y_pred)}")
    print("-" * 50)