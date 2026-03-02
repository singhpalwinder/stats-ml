from ucimlrepo import fetch_ucirepo 
# import pandas as pd


# # fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 

pd = breast_cancer_wisconsin_diagnostic['data']['original']

pd.to_csv('breast_cancer_data.csv', index=False)


