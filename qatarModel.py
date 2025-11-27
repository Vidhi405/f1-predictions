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
    session = fastf1.get_session(year, gp_name, "R")
    session.load()
    df = session.results[[
        "DriverNumber",
        "Abbreviation",
        "TeamName",
        "Position",
        "GridPosition",
        "Status"
    ]].copy()
    df["Year"] = year
    df["GP"] = gp_name
    df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
    df = df.dropna(subset=["Position"])
    return df

def season_results(year):
    schedule = fastf1.get_event_schedule(year)
    races = []
    for _, row in schedule.iterrows():
        if str(row.get("EventFormat", "")).lower() == "testing":
            continue
        try:
            races.append(race_result(year, row["EventName"]))
        except:
            pass
    if not races:
        return None
    return pd.concat(races, ignore_index=True)

def build_all_data(years):
    all_data = []
    for y in years:
        try:
            season_df = season_results(y)
            if season_df is not None:
                all_data.append(season_df)
        except:
            pass
    df = pd.concat(all_data, ignore_index=True)
    return df

def add_driver_form_features(df, window=5):
    df = df.sort_values(["Year", "GP"])
    df["points_finish"] = (df["Position"] <= 10).astype(int)
    df["driver_recent_points_rate"] = (
        df.groupby("Abbreviation")["points_finish"]
          .transform(lambda s: s.shift().rolling(window).mean())
    )
    df["driver_recent_avg_pos"] = (
        df.groupby("Abbreviation")["Position"]
          .transform(lambda s: s.shift().rolling(window).mean())
    )
    df = df.dropna(subset=["driver_recent_points_rate", "driver_recent_avg_pos"])
    return df

def train_models(df):
    y = df["points_finish"]
    feature_columns = [
        "GridPosition",
        "Year",
        "driver_recent_points_rate",
        "driver_recent_avg_pos",
        "Abbreviation",
        "TeamName",
    ]
    X = df[feature_columns].copy()
    numeric_features = [
        "GridPosition",
        "Year",
        "driver_recent_points_rate",
        "driver_recent_avg_pos",
    ]
    categorical_features = ["Abbreviation", "TeamName"]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    tree_clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ]
    )
    tree_clf.fit(X_train, y_train)
    tree_preds = tree_clf.predict(X_test)
    tree_proba = tree_clf.predict_proba(X_test)[:, 1]
    print("Decision Tree Accuracy:", accuracy_score(y_test, tree_preds))
    print("Decision Tree AUC:", roc_auc_score(y_test, tree_proba))
    rf_clf = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                random_state=42,
                n_jobs=-1,
            )),
        ]
    )
    rf_clf.fit(X_train, y_train)
    rf_preds = rf_clf.predict(X_test)
    rf_proba = rf_clf.predict_proba(X_test)[:, 1]
    print("Random Forest Accuracy:", accuracy_score(y_test, rf_preds))
    print("Random Forest AUC:", roc_auc_score(y_test, rf_proba))
    return tree_clf, rf_clf, feature_columns

def get_grid_for_race(year, gp_name):
    try:
        session_q = fastf1.get_session(year, gp_name, "Q")
        session_q.load()
        qres = session_q.results
        return pd.DataFrame({
            "Abbreviation": qres["Abbreviation"],
            "GridPosition": qres["Position"].astype(int),
            "TeamName": qres["TeamName"],
            "Year": year,
        })
    except:
        race_df = race_result(year, gp_name)
        return race_df[["Abbreviation", "TeamName", "GridPosition", "Year"]].copy()

def add_latest_form_to_grid(grid_df, df_form):
    latest = df_form.sort_values(["Year", "GP"]).groupby("Abbreviation").tail(1)[
        ["Abbreviation", "driver_recent_points_rate", "driver_recent_avg_pos"]
    ]
    merged = grid_df.merge(latest, on="Abbreviation", how="left")
    return merged

def predict_race_points_probabilities(year, gp_name, rf_model, df_form, feature_columns):
    grid_df = get_grid_for_race(year, gp_name)
    grid_with_form = add_latest_form_to_grid(grid_df, df_form)
    grid_with_form["driver_recent_points_rate"] = grid_with_form["driver_recent_points_rate"].fillna(0.5)
    grid_with_form["driver_recent_avg_pos"] = grid_with_form["driver_recent_avg_pos"].fillna(10.0)
    X_race = grid_with_form[feature_columns]
    probs = rf_model.predict_proba(X_race)[:, 1]
    grid_with_form["predicted_points_probability"] = probs
    grid_with_form = grid_with_form.sort_values("predicted_points_probability", ascending=False)
    return grid_with_form

if __name__ == "__main__":
    train_years = [2024]
    df = build_all_data(train_years)
    df = add_driver_form_features(df, window=5)
    tree_model, rf_model, feature_cols = train_models(df)
    predict_year = 2024
    predict_gp_name = "Qatar Grand Prix"
    preds = predict_race_points_probabilities(
        predict_year,
        predict_gp_name,
        rf_model,
        df,
        feature_cols
    )
    print(preds[["Abbreviation", "TeamName", "GridPosition", "predicted_points_probability"]])
    print("P1:", preds.iloc[0]["Abbreviation"])
    print("P2:", preds.iloc[1]["Abbreviation"])
    print("P3:", preds.iloc[2]["Abbreviation"])
