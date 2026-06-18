import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def preprocess_data(path):
    raw_data = pd.read_csv(path)
    
    #Applicant_ID
    filled = raw_data["Applicant_ID"].ffill()
    increment = raw_data["Applicant_ID"].isna().groupby(raw_data["Applicant_ID"].notnull().cumsum()).cumsum()
    raw_data["Applicant_ID"] = filled + increment
    
    #Applicant_Income
    conditions_Applicant_Income = [
        (raw_data["Applicant_Income"].isna()) &
        ((raw_data["Employment_Status"].isin(["Salaried", "Self-employed", "Contract"])) |
        (raw_data["Employer_Category"].isin(["Private", "Government", "MNC", "Business"]))),
    
        (raw_data["Applicant_Income"].isna()) &
        ((raw_data["Employment_Status"] == "Unemployed")|
         (raw_data["Employer_Category"] == "Unemployed"))
    ]
    
    choices_Applicant_Income = [
        raw_data["Applicant_Income"].mean(),  
        0  
    ]
    
    raw_data["Applicant_Income"] = np.select(
        conditions_Applicant_Income,
        choices_Applicant_Income,
        default=raw_data["Applicant_Income"]
    )
    
    #Employment_Status
    group_median = raw_data.groupby("Employer_Category")["Applicant_Income"].transform("median")
    
    conditions_Employment_Status = [
        # Unemployed
        (raw_data["Employment_Status"].isna()) &
        ((raw_data["Applicant_Income"] == 0) | (raw_data["Employer_Category"] == "Unemployed")),
    
        # Contract (low income within group)
        (raw_data["Employment_Status"].isna()) &
        (raw_data["Employer_Category"].isin(["Private","Government","MNC"])) &
        (raw_data["Applicant_Income"] < group_median),
    
        # Salaried (remaining in same group)
        (raw_data["Employment_Status"].isna()) &
        (raw_data["Employer_Category"].isin(["Private","Government","MNC"])),
    
        # Self-employed
        (raw_data["Employment_Status"].isna()) &
        (raw_data["Employer_Category"] == "Business"),
    
        #Extra
        (raw_data["Employment_Status"].isna()) &
        (raw_data["Employer_Category"].isna()) &
        (raw_data["Applicant_Income"] > 0)
    ]
    
    choices_Employment_Status = ["Unemployed","Contract","Salaried","Self-employed","Salaried"]
    
    raw_data["Employment_Status"] = np.select(
        conditions_Employment_Status,
        choices_Employment_Status,
        default=raw_data["Employment_Status"]
    )

    #"Coapplicant_Income","Age"
    mean_cols = ["Coapplicant_Income","Age"]
    num_imp = SimpleImputer(strategy="mean")
    raw_data[mean_cols] = num_imp.fit_transform(raw_data[mean_cols])

    #"Marital_Status","Dependents",'Loan_Purpose', 'Education_Level',  'Gender', 'Employer_Category'
    mode_cols = ["Marital_Status","Dependents",'Loan_Purpose', 'Education_Level',  'Gender', 'Employer_Category']
    cat_imp = SimpleImputer(strategy="most_frequent")
    raw_data[mode_cols] = cat_imp.fit_transform(raw_data[mode_cols])

    #"Credit_Score","Existing_Loans","DTI_Ratio",'Savings', 'Collateral_Value', 'Loan_Amount', 'Loan_Term'
    median_cols = ["Credit_Score","Existing_Loans","DTI_Ratio",'Savings', 'Collateral_Value', 'Loan_Amount', 'Loan_Term']
    med_imp = SimpleImputer(strategy="median")
    raw_data[median_cols] = med_imp.fit_transform(raw_data[median_cols])

    #Property_Area
    raw_data['Property_Area'] = raw_data.groupby('Loan_Purpose')['Property_Area'].transform(lambda x: x.fillna(x.mode()[0]))

    raw_data.dropna(inplace=True)

    return raw_data




def encoding(data):
    le1 = LabelEncoder()
    data["Education_Level"] = le1.fit_transform(data["Education_Level"])
    le2 = LabelEncoder()
    data["Loan_Approved"] = le2.fit_transform(data["Loan_Approved"])


    cols = ["Employment_Status", "Marital_Status", "Loan_Purpose", "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = ohe.fit_transform(data[cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cols), index=data.index)
    encoded_data = pd.concat([data.drop(columns=cols), encoded_df], axis=1)

    return encoded_data


def feature_sq(data,new_feature_name,feature):
    data[new_feature_name] = data[feature] ** 2
    