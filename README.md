#  Civil war and internal conflict predictor

## Objectives
- Used event data from the ICEWS dataset to build a model with a ~75% chance to predict a civil war (AUC ~ 0.75)
- Used the PITF data as target variable to predict the start, the ongoingness and the end of civil war. 
- Determined key sectors and how the interact at the beginning of a civil war using t-tests. 
- Used AUC as the measure to optimise given the imbalanced in the target variable (less wars than peace)
- Used a sliding window approach to predict the outcome for the next month. 
- Built a random forest with AUC ~0.75 due to data imbalance and long list of features. 

## Packages
**Python Version:** 3.8.3

**Packages used:** pandas, numpy, scikit-learn, matplotlib, seaborn

## Cleaning the data
- Selected only internal events
- Simplified the sectors into government, opposition, insurgents and people.
- Used events where source and target are different. 
- Simplified the kind of interaction (using just the first two digits of the CAMEO code). 
- Used counts for each kind of interaction between each sector as input variables. 

## Project report
One remarkable thing about the model is that it was able to detect not only civil wars but conflict escalation. For example, it was able to detect conflict escalation in February 2020 in India, where riots in Delhi ended up with 53 deaths. 

A project report with further information can be seen in the [CivilWarForecast.pdf](https://github.com/jorgerodpen/Civil-War-Forecast/blob/main/CivilWarForecast.pdf) document stored in this same repository.