import pandas as pd
import numpy as np
import os
import sys

# Add parent directory containing utils.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from utils import pre_process, save_predictions_to_csv
# Question A

# now build a decision tree to predict the result given the trainingData
# Step 0: 
from decisionTree import DecisionTreeModel
from utils import evaluate, plotA

def main(train_path, val_path, test_path, output_path):
    
    trainData = pd.read_csv(train_path)
    testData = pd.read_csv(test_path)
    validationData = pd.read_csv(val_path)
    
    print(f"Shape of trainData is : {trainData.shape}")
    print(f"Shape of testData is : {testData.shape}")
    print(f"Shape of validationData is : {validationData.shape}")

    # Step 2: Encode the columns["team", "opp", "host", "month"]
    categorical_cols = ["team", "opp", "host", "month"]
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]
    
    trainDataX, trainDataY, validationDataX, validationDataY, testDataX, testDataY =  pre_process(trainData, validationData, testData, categorical_cols = categorical_cols, continuous_cols = continuous_cols)
    team_summary = trainData.groupby(["team","opp", "host", "toss","result"]).size().unstack(fill_value=0)

    maxDepths = [5, 10, 15, 20]
    trainAccuracies = []
    validationAccuracies = []
    testAccuracies = []
    
    categorical_cols = ["team", "opp", "host", "month", "toss", "day_match", "bat_first", "format"]
    continuous_cols = ["Unnamed", "year", "fow", "score", "rpo"]

    best_model = None
    best_validation_accuracy = 0
    best_max_depth = 0
    
    for maxDepth in maxDepths:
        print(f"---------------Testing for maxDepth : {maxDepth}-------------------")
        model = DecisionTreeModel(maxDepth, categorical_cols = categorical_cols, continuous_cols = continuous_cols)
    
        # TRAIN THE MODEL WITH TRAIN DATA
        print("Training on the train data")
        model.fit(trainDataX, trainDataY)
    
        # PREDICT THE MODEL ON TRAIN DATA
        print("Prediction on train data")
        predictY_train = model.predict(trainDataX)
    
        # EVALUATION
        train_accuracy = evaluate(predictY_train, trainDataY)
    
        # PREDICT THE MODEL ON VALIDATION DATA
        print("Prediction on validation data")
        predictY_validation = model.predict(validationDataX)
    
        # EVALUATION
        validation_accuracy = evaluate(predictY_validation, validationDataY)
    
        # PREDICT THE MODEL ON Test DATA
        print("Prediction on test data")
        predictY_test = model.predict(testDataX)
    
        # EVALUATION
        test_accuracy = evaluate(predictY_test, testDataY)
    
        trainAccuracies.append(train_accuracy)
        validationAccuracies.append(validation_accuracy)
        testAccuracies.append(test_accuracy)

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model = model
            best_max_depth = maxDepth
    
    plotA(maxDepths, trainAccuracies, validationAccuracies, testAccuracies, "plots/decisionTreeDepthVsAccuracy.png")

    print(f"Best validation accuracy is coming to be {best_validation_accuracy} and for the depth {best_max_depth}")
    best_predictY_test = best_model.predict(testDataX)
    save_predictions_to_csv(best_predictY_test, output_path)

if __name__ == "__main__":
    train_path = sys.argv[1]
    val_path = sys.argv[2]
    test_path = sys.argv[3]
    output_path = sys.argv[4]
    main(train_path, val_path, test_path, output_path)

