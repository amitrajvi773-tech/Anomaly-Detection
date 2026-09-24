# ANOMALY DETECTION USING MACHINE LEARNING
# SMOTETomek + MLflow

import warnings

import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier


# Ignore unnecessary warnings
warnings.filterwarnings("ignore")


# STEP 1: CREATE DATASET

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=2,
    n_redundant=8,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=42
)

print("Original Dataset")
print("----------------")

unique, counts = np.unique(y, return_counts=True)

for class_value, count in zip(unique, counts):
    print(f"Class {class_value}: {count}")

print()


# STEP 2: TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

print("Training data size:", len(X_train))
print("Testing data size :", len(X_test))
print()


# STEP 3: HANDLE CLASS IMBALANCE

smote_tomek = SMOTETomek(random_state=42)

X_train_resampled, y_train_resampled = smote_tomek.fit_resample(
    X_train,
    y_train
)

print("After SMOTETomek")
print("----------------")

unique, counts = np.unique(
    y_train_resampled,
    return_counts=True
)

for class_value, count in zip(unique, counts):
    print(f"Class {class_value}: {count}")

print()


# STEP 4: DEFINE MODELS


models = [

    {
        "name": "Logistic Regression",
        "model": LogisticRegression(
            C=1,
            solver="liblinear"
        ),
        "train_X": X_train,
        "train_y": y_train
    },

    {
        "name": "Random Forest",
        "model": RandomForestClassifier(
            n_estimators=30,
            max_depth=3,
            random_state=42
        ),
        "train_X": X_train,
        "train_y": y_train
    },

    {
        "name": "XGBoost",
        "model": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
        "train_X": X_train,
        "train_y": y_train
    },

    {
        "name": "XGBoost With SMOTETomek",
        "model": XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ),
        "train_X": X_train_resampled,
        "train_y": y_train_resampled
    }

]

# OPTION 1:
# Use local MLflow server

  





# If you have your own MLflow server, replace the above with:
#
# mlflow.set_tracking_uri(
#     "http://54.152.148.165:5000"
# )


# STEP 6: TRAIN MODELS + EVALUATE + TRACK WITH MLFLOW

results = []


for item in models:

    model_name = item["name"]
    model = item["model"]

    train_X = item["train_X"]
    train_y = item["train_y"]


    print("=" * 60)
    print(model_name)
    print("=" * 60)


    # Start MLflow run

    with mlflow.start_run(run_name=model_name):

        # Train model

        model.fit(
            train_X,
            train_y
        )


        # Prediction

        y_pred = model.predict(X_test)


        # Classification report

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True
        )


        # Extract metrics

        accuracy = report["accuracy"]

        recall_class_0 = report["0"]["recall"]

        recall_class_1 = report["1"]["recall"]

        f1_macro = report["macro avg"]["f1-score"]


        # Print results

        print(f"Accuracy       : {accuracy:.4f}")

        print(
            f"Recall Class 0 : {recall_class_0:.4f}"
        )

        print(
            f"Recall Class 1 : {recall_class_1:.4f}"
        )

        print(
            f"F1 Macro       : {f1_macro:.4f}"
        )

        print()


        # Log parameters

        params = model.get_params()

        mlflow.log_params(params)


        # Log metrics

        mlflow.log_metrics({

            "accuracy": accuracy,

            "recall_class_0": recall_class_0,

            "recall_class_1": recall_class_1,

            "f1_score_macro": f1_macro

        })


        # Log model

        if "XGBoost" in model_name:

            mlflow.xgboost.log_model(
                model,
                name="model"
            )

        else:

            mlflow.sklearn.log_model(
                model,
                name="model"
            )


        # Store result

        results.append({

            "Model": model_name,

            "Accuracy": accuracy,

            "Recall Class 0": recall_class_0,

            "Recall Class 1": recall_class_1,

            "F1 Score": f1_macro

        })


# STEP 7: DISPLAY FINAL RESULTS

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for result in results:

    print()
    print(result["Model"])

    print(
        f"Accuracy       : {result['Accuracy']:.4f}"
    )

    print(
        f"Recall Class 0 : {result['Recall Class 0']:.4f}"
    )

    print(
        f"Recall Class 1 : {result['Recall Class 1']:.4f}"
    )

    print(
        f"F1 Score       : {result['F1 Score']:.4f}"
    )