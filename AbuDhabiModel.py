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

def season(year):
    schedule=fastf1.get_event_schedule(year)
    races=[]

    for _, row in schedule.iterrows():
        if str(row.get("EventFormat","")).lower()=="testing":
            continue

        gp_name=row["EventName"]
        round=row.get("RoundNumber",None)

        try:
            session=fastf1.get_session(year,gp_name,"R")
            session.load()

            df=session.results[["DriverNumber","Abbreviation","TeamName","Position","GridPosition","Status"]].copy()

            df["Year"]=year
            df["GP"]=gp_name
            df["Round"]=round
            df["Position"]=pd.to_numeric(df["Position"],errors="coerce")
            df=df.dropna(subset=["Position"])
            races.append(df)
        except Exception as e:
            pass

    if not races:
        return None
    
    return pd.concat(races, ignore_index=True)


def databuild(years):
    all=[]
    for i in years:
        try:
            season_df=season(i)
            if season_df is not None:
                all.append(season_df)

        except Exception:
            pass
    df=pd.concat(all,ignore_index=True)

    return df


def driverform(df, window=5):
    df=df.sort_values(["Year","Round"])
    df["points_finish"]=(df["Position"]<=10).astype(int)
    df["driver_recent_points_rate"]=(df.groupby("Abbreviation")["points_finish"].transform(lambda s: s.shift().rolling(window).mean()))
    df["driver_recent_avg_pos"]=(df.groupby("Abbreviation")["Position"].transform(lambda s: s.shift().rolling(window).mean()))

    return df

def drivertrackform(df,window=3):
    df=df.sort_values(["Year","Round"])

    df["driver_track_points_rate"] = (df.groupby(["Abbreviation", "GP"])["points_finish"].transform(lambda s: s.shift().rolling(window).mean()))
    df["driver_track_avg_pos"] = (df.groupby(["Abbreviation", "GP"])["Position"].transform(lambda s: s.shift().rolling(window).mean()))
    return df



def teamform(df,window=5):
    df=df.sort_values(["Year","Round"])
    df["team_recent_avg_pos"]=(df.groupby("TeamName")["Position"].transform(lambda s: s.shift().rolling(window).mean()))

    return df


def finalfeatures(df):
    df=driverform(df,window=5)
    df= drivertrackform(df,window=3)
    df= teamform(df,window=5)
    df["RoundNorm"]=df["Round"].astype(float)
    df=df.dropna(subset=["driver_recent_points_rate","driver_recent_avg_pos","team_recent_avg_pos"])

    return df


def model(df):
    y=df["points_finish"]

    feature_columns=["GridPosition","Year","RoundNorm","driver_recent_points_rate","driver_recent_avg_pos","driver_track_points_rate","driver_track_avg_pos","team_recent_avg_pos","Abbreviation","TeamName"]

    X=df[feature_columns].copy()

    X["driver_track_points_rate"]=X["driver_track_points_rate"].fillna(0.5)
    X["driver_track_avg_pos"]=X["driver_track_avg_pos"].fillna(10.0)
    X["team_recent_avg_pos"]=X["team_recent_avg_pos"].fillna(10.0)

    numeric_features=["GridPosition","Year","RoundNorm","driver_recent_points_rate","driver_recent_avg_pos","driver_track_avg_pos","team_recent_avg_pos"]
    categorical_features=["Abbreviation","TeamName"]

    preprocess=ColumnTransformer(transformers=[("num","passthrough",numeric_features),("cat",OneHotEncoder(handle_unknown="ignore"),categorical_features),])

    X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    tree_clf=Pipeline(steps=[("preprocess",preprocess),("classifier",DecisionTreeClassifier(max_depth=5, random_state=42)),])
    tree_clf.fit(X_train,y_train)
    tree_preds=tree_clf.predict(X_test)
    tree_prob=tree_clf.predict_proba(X_test)[:,1]
    print("Decision Tree Accuracy: ",accuracy_score(y_test,tree_preds))
    print("Decision Tree AUC: ",roc_auc_score(y_test, tree_prob))

    rf_clf=Pipeline(steps=[("preprocess",preprocess),("model",RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1,class_weight="balanced",)),])

    rf_clf.fit(X_train,y_train)
    rf_preds=rf_clf.predict(X_test)
    rf_prob=rf_clf.predict_proba(X_test)[:, 1]

    print("Random Forest Accuracy: ",accuracy_score(y_test, rf_preds))
    print("Random Forest AUC: ",roc_auc_score(y_test, rf_prob))

    return tree_clf, rf_clf, feature_columns


def grid(year, gp_name):
    schedule=fastf1.get_event_schedule(year)
    row=schedule.loc[schedule["EventName"]==gp_name]
    if not row.empty:
        round_number=row.iloc[0].get("RoundNumber",None)
    else:
        round_number=None

    try:
        sessionq=fastf1.get_session(year,gp_name,"Q")
        sessionq.load()

        qres=sessionq.results
        grid_df=pd.DataFrame({"Abbreviation": qres["Abbreviation"],"GridPosition": qres["Position"].astype(int),"TeamName":qres["TeamName"],"Year":year,})

    except Exception:
        race_df=season(year)
        race_df=race_df[race_df["GP"]==gp_name].copy()
        grid_df=race_df[["Abbreviation","TeamName","GridPosition","Year"]].copy()
    grid_df["RoundNorm"] = float(round_number) if round_number is not None else np.nan
    return grid_df

def latestgrid(grid_df,df_form,gp_name):
    df_form=df_form.sort_values(["Year","Round"])
    overall_latest=df_form.groupby("Abbreviation").tail(1)[["Abbreviation","driver_recent_points_rate","driver_recent_avg_pos","team_recent_avg_pos",]]
    
    trackhistory=df_form[df_form["GP"]==gp_name].copy()
    tracklatest=trackhistory.sort_values(["Year","Round"]).groupby("Abbreviation").tail(1)[["Abbreviation","driver_track_points_rate","driver_track_avg_pos",]]
    merged=grid_df.merge(overall_latest, on="Abbreviation",how="left")
    merged = merged.merge(tracklatest, on="Abbreviation", how="left")

    return merged


def predictProbabilities(year, gp_name, rf_model, df_form, feature_columns):
    grid_df =grid(year, gp_name)
    grid_with_form =latestgrid(grid_df, df_form, gp_name)

    grid_with_form["driver_recent_points_rate"] = grid_with_form["driver_recent_points_rate"].fillna(0.5)
    grid_with_form["driver_recent_avg_pos"] = grid_with_form["driver_recent_avg_pos"].fillna(10.0)
    grid_with_form["driver_track_points_rate"] = grid_with_form["driver_track_points_rate"].fillna(0.5)
    grid_with_form["driver_track_avg_pos"] = grid_with_form["driver_track_avg_pos"].fillna(10.0)
    grid_with_form["team_recent_avg_pos"] = grid_with_form["team_recent_avg_pos"].fillna(10.0)
    grid_with_form["RoundNorm"] = grid_with_form["RoundNorm"].fillna(grid_with_form["RoundNorm"].median())

    X_race = grid_with_form[feature_columns].copy()

    for col in ["driver_track_points_rate", "driver_track_avg_pos", "team_recent_avg_pos"]:
        if col in X_race.columns:
            X_race[col] = X_race[col].fillna(0 if "rate" in col else 10.0)

    probs = rf_model.predict_proba(X_race)[:, 1]
    grid_with_form["predicted_points_probability"] = probs
    grid_with_form = grid_with_form.sort_values("predicted_points_probability", ascending=False)
    return grid_with_form


if __name__ == "__main__":
    train_years = [2024, 2025]

    print("Building dataset...")
    df =databuild(train_years)
    df = finalfeatures(df)

    print("Training models...")
    tree_model, rf_model, feature_cols = model(df)

    predict_year = 2025
    predict_gp_name = "Abu Dhabi Grand Prix"

    print(f"\nPredicting for {predict_year} {predict_gp_name}...")
    preds = predictProbabilities(predict_year,predict_gp_name,rf_model,df,feature_cols)

    print(preds[["Abbreviation", "TeamName", "GridPosition", "predicted_points_probability"]])
    print("P1:", preds.iloc[0]["Abbreviation"])
    print("P2:", preds.iloc[1]["Abbreviation"])
    print("P3:", preds.iloc[2]["Abbreviation"])