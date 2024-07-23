import pandas as pd
import os
from .load import load_file
from sklearn.impute import KNNImputer
import datetime
from sklearn.preprocessing import LabelEncoder

def remove_rent(df, sale_type='residential_sale'):
    return df[df['TypeOfSale'] == sale_type]

def strip_data(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
    return df

def no_duplicates(df): 
    df_unique = df.drop(columns=['Url', 'PropertyId', 'SubtypeOfProperty']).drop_duplicates()
    return df.loc[df_unique.index]

def remove_outliers(df, column, multiplier=2):
    Q1 = df[column].quantile(0.20)
    Q3 = df[column].quantile(0.80)
    IQR = Q3 - Q1
    
    print(f"Q1 (20th percentile): {Q1} Q3 (80th percentile): {Q3} IQR: {IQR}")
    
    before_count = df.shape[0]
    
    if IQR == 0:
        lower_bound = Q1
        upper_bound = Q3 
    else:
        lower_bound = max(Q1 - multiplier * IQR, 0)
        upper_bound = Q3 + multiplier * IQR

    df = df[~((df[column] < lower_bound) | (df[column] > upper_bound))]

    print(f"Lower bound: {lower_bound} Upper bound: {upper_bound}")

    after_count = df.shape[0]
    print(f"{column}: Removed {before_count - after_count} outliers")
    return df

def select_features(df):
    df = df[
        ((df["PEB"].isin(["A++", "A+", "A", "B", "C", "D", "E", "F"])) | (df['PEB'].isna())) &
        (((df['PostalCode'] >= 1000) & (df['PostalCode'] <= 9999)) | (df['PostalCode'].isna())) &
        ((df['ConstructionYear'] <= datetime.datetime.now().year + 20) | (df['ConstructionYear'].isna()))
    ][[
        "Price", "PostalCode", "BathroomCount", "BedroomCount", "ConstructionYear", "NumberOfFacades", "PEB", 
        "SurfaceOfPlot", "LivingArea", "GardenArea", "StateOfBuilding", "SwimmingPool", "Terrace", "ToiletCount", "RoomCount",
        "Fireplace", "Furnished", "Kitchen", "FloodingZone"
    ]]

    df['LivingArea_per_Bedroom'] = df['LivingArea'] / (df['BedroomCount'] + 1)
    df['GardenArea_per_Bedroom'] = df['GardenArea'] / (df['BedroomCount'] + 1)
    df['PropertyAge'] = datetime.datetime.now().year - df['ConstructionYear']
    df['LivingArea_to_TotalArea'] = df['LivingArea'] / (df['SurfaceOfPlot'] + 1)
    df['Bedroom_to_Facades'] = df['BedroomCount'] / (df['NumberOfFacades'] + 1)
    df['FloodRisk'] = df['FloodingZone'].apply(lambda x: 1 if x != 'NON_FLOOD_ZONE' else 0)

    high_quality_kitchens = ['HYPER_EQUIPPED', 'SUPER_EQUIPPED', 'USA_INSTALLED']
    df['HighQualityKitchen'] = df['Kitchen'].apply(lambda x: 1 if x in high_quality_kitchens else 0)

    df.drop(columns=['FloodingZone'], inplace=True)
    df.drop(columns=['Kitchen'], inplace=True)
    df.drop(columns=['ConstructionYear'], inplace=True)

    print(f"Rows after initial selection: {df.shape[0]}")

    multi_dico = {
        "Price": 5,
        "BedroomCount": 3,
        "BathroomCount": 3,
        "NumberOfFacades": 2,
        "SurfaceOfPlot": 4,
        "LivingArea": 4,
        "GardenArea": 4,
        "ToiletCount": 2,
    }

    print("===============")
    for k, i in multi_dico.items():
        df = remove_outliers(df, k, multiplier=i)
    print("===============")

    return df

def saveNan(df):
    output_path = '../data/cleaned_data.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"DataFrame modifié enregistré sous '{output_path}'")

def clean_nan(df, n_neighbors=5):
    print("clean NA")

    df["SwimmingPool"].fillna(0, inplace=True)
    df["Terrace"].fillna(0, inplace=True)
    df["GardenArea"].fillna(0, inplace=True)
    df["Fireplace"].fillna(0, inplace=True)
    df["Furnished"].fillna(0, inplace=True)

    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le
    
    numeric_df = df.select_dtypes(include=['number'])
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed = imputer.fit_transform(numeric_df)
    numeric_df_imputed = pd.DataFrame(df_imputed, columns=numeric_df.columns, index=numeric_df.index)
    df.update(numeric_df_imputed)
    return df

def clean(df):
    df = remove_rent(df)
    df = strip_data(df)
    df = no_duplicates(df)
    df = select_features(df)
    df = clean_nan(df)
    return df

if __name__ == "__main__":
    df = load_file("../data/final_dataset.json")
    if df is False:
        print("Failed to load file.")
    
    df = clean(df)
    saveNan(df)
