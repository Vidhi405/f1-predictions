import fastf1
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

fastf1.Cache.enable_cache("f1_cache")

def race_result(year, gp_name):
    session=fastf1.get_session(year, gp_name, "R")
    session.load()

    df=session.results[[{"DriverNumber", "Abbreviation","TeamName","Position","GridPosition","Status"}]].copy()
    df["Year"]=year
    df["GP"]=gp_name
    return df


def season_results(year):
    schedule=fastf1.get_event_schedule(year)
    races=[]
    for _, row in schedule.iterrows():
        if row["EventFormat"]=="testing":
            continue
        try:
            races.append(race_result(year,row["EventName"]))
        except:
            pass

    if not races:
        return none
    return pd.concat(races, ignore_index=True)


def build_all_data(years):
    all_data=[]
    for y in [2021,2022,2023,2024]:
        try:
            all_data.append(season_results(y))
        except:
            pass

    df=pd.concat(all_data, ignore_index=True)
    return df

def add_driver_form_features(df,window=5):
    df=df.sort_value(["Year","GP"])
    df["points_finish"]=(df["Position"]<=10).astype(int)
    df["driver_recent_points_rate"]=(df.groupby("Abbreviation")["points_finish"].apply(lamba s: s.shift().rolling(window).mean()))
    df["driver_recent_avg_pos"]=(df.groupby("Abbreviation")["Position"].apply(lambda s: s.shift().rolling(window).mean()))

    df=df.dropna(subset=["driver_recent_points_rate","driver_recent_avg_pos"])
    return df

def train_model(df):
    y=df["points_finish"]

    feature_columns=["GridPosition","Year","driver_recent_points_rate","driver_recent_avg_pos","Abbreviation","TeamName"]
    
    X=df[feature_columns].copy()

    numeric_features=["GridPosition","Year","driver_recent_points_rate","driver_recent_avg_pos"]
    categorical_features=["Abbreviation","TeamName"]
    preprocess=ColumnTransformer(transformers=[("num","passthrough",numeric_features),("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features),])
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=42, statify=y)
    tree_clf=Pipeline(steps=[("preprocess",preprocess),("model",DecisionTreeClassifier(max_depth=5, random_state=42)),])
    tree_clf.fit(X_train, y_train)
    tree_preds=tree_clf.predict(X_test)
    tree_proba=tree_clf.predict_proba(X_test)[:, 1]
    print("Decision Tree Accuracy:", accuracy_score(y_test, tree_preds))
    print("Decision Tree AUC:", roc_auc_score(y_test, tree_proba))