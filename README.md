# World Cup 2022 Participation Prediction Model

## Project Overview

This repository contains a machine learning project designed to predict whether a national soccer team participated in the **2022 FIFA World Cup**. By leveraging a custom-built dataset of team performance metrics, historical data, and global rankings, the model identifies key indicators that lead to tournament qualification.

## Dataset Analysis

The model is trained on a unique dataset aggregating information for 46 national teams.


Features included:


* `Rank` & `TotalPoints`: Official FIFA World Ranking data.
* `Confederation`: Geographic region (UEFA, CONMEBOL, CAF, etc.).
* `NumberLocalFinals`: Number of appearances in continental tournament finals (e.g., Copa América, UEFA Euro).
* `WorldCupParticipations`: Total historical appearances in the FIFA World Cup.
* `AVG`: Average goals scored per match across the season.
* `Big5`: Count of players active in the Big 5 European Leagues (FBref data).

Data Sources: * FIFA World Rankings.


* FBref (European League Nationalities).
* Historical match databases and continental competition statistics.



## Technical Implementation

* **Language:** Python.
* **Libraries:** Pandas, Scikit-Learn, NumPy, Matplotlib.
* **Machine Learning Pipeline:**
* **Preprocessing:** Handled categorical data using `OneHotEncoder` and scaled target variables with `LabelEncoder`.
* **Algorithm:** Implemented a `DecisionTreeClassifier` to map out the logic of qualification.
* **Validation:** Used `RepeatedKFold` (10x10-fold cross-validation) to ensure model stability across different data subsets.



## Model Performance

The decision tree model provided the following results:

* **Mean Accuracy:**  (63%).
  
* **Standard Deviation:** (0.186).

**Insights:** Initial analysis suggests that `TotalPoints` and historical `WorldCupParticipations` are the strongest early indicators in the decision tree branches.



## Visualizations

The visual decision-making logic of the tree can be found in the repository:

*  `MyDataTree.png`: Displays the primary split points (e.g., `TotalPoints <= 1519.055`) and the resulting classifications.



## Repository Contents

* `Lab1_MY_Data.py`: The full Python script for data processing and modeling.
* `MyData1.arff`: The structured dataset in ARFF format.
* `MBLab1.pdf`: A comprehensive report detailing the methodology and cross-validation results.
