import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments from command
parser = argparse.ArgumentParser()
parser.add_argument("--nn", type=int, required=False, default=5)
args = parser.parse_args()

# evaluation function
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the iris csv file from local
    data = pd.read_csv("data/iris.csv")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    # The predicted column is "variety" which is a categorical variable
    train_x = train.drop(["variety"], axis=1)
    test_x = test.drop(["variety"], axis=1)
    train_y = train[["variety"]]
    test_y = test[["variety"]]

    n_neighbors = args.nn

    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # backend store: params/metrics/tags

    experiment_name = "iris_experiment"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location="./mlruns",  # artifact store: models/files
            tags={"version": "1.0", "type": "classification"},
        )

    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    get_exp = mlflow.get_experiment(exp_id)
    
    print("Name: {}".format(get_exp.name))
    print("Experiment_id: {}".format(get_exp.experiment_id))
    print("Artifact Location: {}".format(get_exp.artifact_location))
    print("Tags: {}".format(get_exp.tags))
    print("Lifecycle_stage: {}".format(get_exp.lifecycle_stage))
    print("Creation timestamp: {}".format(get_exp.creation_time))
    
    exp = mlflow.set_experiment(experiment_name)

    with mlflow.start_run(experiment_id=exp.experiment_id) as run:
        tags = {"release_version": "0.1",
                "model_type": "KNNClassifier",
                "dataset": "iris"}
        mlflow.set_tags(tags)
        knn = KNeighborsClassifier(n_neighbors=int(n_neighbors))
        knn.fit(train_x, train_y)

        predicted_qualities = knn.predict(test_x)

        accuracy = knn.score(test_x, test_y)
        precision = np.mean(predicted_qualities == test_y.values.flatten())
        recall = np.mean(predicted_qualities == test_y.values.flatten())

        print("KNN model (n_neighbors={:d}):".format(knn.n_neighbors))
        print("  Accuracy: %s" % accuracy)
        print("  Precision: %s" % precision)
        print("  Recall: %s" % recall)


        mlflow.log_params({"n_neighbors": knn.n_neighbors})
        mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall})

        mlflow.log_artifacts("data/")
        mlflow.sklearn.log_model(
            knn,
            name="mymodel",
            skops_trusted_types=[
                "sklearn.metrics._dist_metrics.EuclideanDistance64",
                "sklearn.neighbors._kd_tree.KDTree",
            ],
        )

        run = mlflow.active_run()
        print("Active run_id: {}".format(run.info.run_id))

        print("Artifact URI: {}".format(mlflow.get_artifact_uri()))

    latest_run = mlflow.search_runs(experiment_ids=exp.experiment_id).iloc[0]
    print("Latest run_id: {}".format(latest_run.run_id))
