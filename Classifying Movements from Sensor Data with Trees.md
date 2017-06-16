Summary
-------

Data from wearable devices accelerometers on the belt, forearm, arm, and
dumbell of 6 participants asked to perform weight lifting correctly
(classe A) and incorrectly in 4 different ways (classe B, C, D and E)
are used to build a model able to predict the way the exercise was
performed. More in formation on the data is available on
<http://groupware.les.inf.puc-rio.br/har>.

Decision trees are chosen as the machine-learning models, for
classification. A Simple Decision Tree is fitted first. The tree turns
out to be fast but not very accurate. To improve accuracy many trees are
also fitted with Random Forests that is good for beginners, robust to
over-fitting and usually yields very accurate non-linear models. The
random forests model is finally applied to predict the way variable
(classe A, B, C, D, E) for 20 test cases provided.

Getting and Cleaning the Data
-----------------------------

Loading packages.

    library(downloader); library(lattice); library(ggplot2); library(caret); library(rpart); library(rattle); library(rpart.plot); library(ranger); library(e1071)

    ## Rattle: A free graphical interface for data mining with R.
    ## Version 4.1.0 Copyright (c) 2006-2015 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

Downloading and importing the training and testing data sets.

    url_train <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    url_val <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    file_train <- "pml-training.csv"
    file_val <- "pml-testing.csv"
    download(url_train, destfile = file_train)
    download(url_val, destfile = file_val)
    train <- read.csv(file_train, na.strings = c("NA", ""))
    val <- read.csv(file_val, na.strings = c("NA", ""))

The train (training) and the val (20 test cases) sets include 160
variables including the "classe" on the train set that we want to
predict on the val set for the "problem\_id" variable.

    dim(train); dim(val)

    ## [1] 19622   160

    ## [1]  20 160

    names(train)[160]; names(val)[160]

    ## [1] "classe"

    ## [1] "problem_id"

The first seven columns on both sets include variables that not seem to
be very relevant.

    head(train[, 1:7], 1)

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
    ##   new_window num_window
    ## 1         no         11

    head(val[, 1:7], 1)

    ##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
    ## 1 1     pedro           1323095002               868349 05/12/2011 14:23
    ##   new_window num_window
    ## 1         no         74

So the first seven columns from both data sets are removed.

    train_data <- train[, -(1:7)]
    val_data <- val[, -(1:7)]

Of the left columns, 100 include NAs.

    sum(colSums(is.na(train_data)) > 0); sum(colSums(is.na(val_data)) > 0)

    ## [1] 100

    ## [1] 100

In all of them NAs represent more 95% of the cases.

    sum((colSums(is.na(train_data))/(dim(train_data)[1])) >= 0.95); sum((colSums(is.na(val_data))/(dim(val_data)[1])) >= 0.95)

    ## [1] 100

    ## [1] 100

So instead of inputing values, all the columns with variables that
include any NAs are removed.

    train_data <- train_data[, colSums(is.na(train_data)) == 0]
    val_data <- val_data[, colSums(is.na(val_data)) == 0]

Splitting the Data
------------------

The cleaned training set is split in two (70:30) to train models for
prediction and calculate out of sample error respectively.

    set.seed(2266)
    inTrain <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
    training <- train_data[inTrain, ]
    testing <- train_data[-inTrain, ]

Model Fitting Using Decision Trees and Random Forest
----------------------------------------------------

Two models are trained with the training data set using simple decision
trees and random forests respectively.

### Simple Decision Tree

We start with a simple decision tree without variables transformation.
The model turns out to be fast but not very accurate (0.52).

    fit_trees <- train(classe ~., data = training, method = "rpart")
    print(fit_trees)

    ## CART 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Bootstrapped (25 reps) 
    ## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   cp          Accuracy   Kappa     
    ##   0.03387244  0.5223828  0.38028359
    ##   0.05970908  0.3934496  0.16834688
    ##   0.11595972  0.3351582  0.07501235
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was cp = 0.03387244.

Plot of the fitted tree.

    fancyRpartPlot(fit_trees$finalModel)

![](Classifying_Movements_from_Sensor_Data_with_Trees_files/figure-markdown_strict/unnamed-chunk-11-1.png)

The model is tested out sample and the confusion matrix is generated
confirming low accuracy (0.49).

    predict_trees <- predict(fit_trees, testing)
    confusionMatrix(testing$classe, predict_trees)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1527   24  117    0    6
    ##          B  486  355  298    0    0
    ##          C  445   33  548    0    0
    ##          D  427  169  368    0    0
    ##          E  149  151  299    0  483
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.495           
    ##                  95% CI : (0.4821, 0.5078)
    ##     No Information Rate : 0.5155          
    ##     P-Value [Acc > NIR] : 0.9992          
    ##                                           
    ##                   Kappa : 0.3405          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.5033  0.48497  0.33620       NA  0.98773
    ## Specificity            0.9484  0.84786  0.88766   0.8362  0.88899
    ## Pos Pred Value         0.9122  0.31168  0.53411       NA  0.44640
    ## Neg Pred Value         0.6421  0.92056  0.77732       NA  0.99875
    ## Prevalence             0.5155  0.12438  0.27698   0.0000  0.08309
    ## Detection Rate         0.2595  0.06032  0.09312   0.0000  0.08207
    ## Detection Prevalence   0.2845  0.19354  0.17434   0.1638  0.18386
    ## Balanced Accuracy      0.7259  0.66641  0.61193       NA  0.93836

### Random Forest Model

To improve accuracy, many trees are fitted to bootsrapped samples of the
training data randomly sampling columns at each split. Default values
for hyper parameters are used except for a control element with 5 folds
for cross validation, half the default value, to reduce computation
time. The “ranger” package is chosen over the “rf” because is more than
two times faster.

    library(caret)
    myControl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
    fit_rf_ranger <- train(classe ~., data = training,
                     method = "ranger", trControl = myControl)

    ## + Fold1: mtry= 2 
    ## - Fold1: mtry= 2 
    ## + Fold1: mtry=27 
    ## - Fold1: mtry=27 
    ## + Fold1: mtry=52 
    ## Growing trees.. Progress: 81%. Estimated remaining time: 7 seconds.
    ## - Fold1: mtry=52 
    ## + Fold2: mtry= 2 
    ## - Fold2: mtry= 2 
    ## + Fold2: mtry=27 
    ## - Fold2: mtry=27 
    ## + Fold2: mtry=52 
    ## Growing trees.. Progress: 84%. Estimated remaining time: 5 seconds.
    ## - Fold2: mtry=52 
    ## + Fold3: mtry= 2 
    ## - Fold3: mtry= 2 
    ## + Fold3: mtry=27 
    ## - Fold3: mtry=27 
    ## + Fold3: mtry=52 
    ## Growing trees.. Progress: 88%. Estimated remaining time: 4 seconds.
    ## - Fold3: mtry=52 
    ## + Fold4: mtry= 2 
    ## - Fold4: mtry= 2 
    ## + Fold4: mtry=27 
    ## - Fold4: mtry=27 
    ## + Fold4: mtry=52 
    ## Growing trees.. Progress: 78%. Estimated remaining time: 8 seconds.
    ## - Fold4: mtry=52 
    ## + Fold5: mtry= 2 
    ## - Fold5: mtry= 2 
    ## + Fold5: mtry=27 
    ## - Fold5: mtry=27 
    ## + Fold5: mtry=52 
    ## Growing trees.. Progress: 88%. Estimated remaining time: 4 seconds.
    ## - Fold5: mtry=52 
    ## Aggregating results
    ## Selecting tuning parameters
    ## Fitting mtry = 27 on full training set

    print(fit_rf_ranger)

    ## Random Forest 
    ## 
    ## 13737 samples
    ##    52 predictor
    ##     5 classes: 'A', 'B', 'C', 'D', 'E' 
    ## 
    ## No pre-processing
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 10988, 10990, 10990, 10990, 10990 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  Accuracy   Kappa    
    ##    2    0.9903912  0.9878440
    ##   27    0.9914102  0.9891337
    ##   52    0.9842763  0.9801082
    ## 
    ## Accuracy was used to select the optimal model using  the largest value.
    ## The final value used for the model was mtry = 27.

The model is tested out sample and the confusion matrix is generated
confirming high accuracy (0.99).

    predict_rf_ranger <- predict(fit_rf_ranger, testing)
    confusionMatrix(testing$classe, predict_rf_ranger)

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1673    1    0    0    0
    ##          B   15 1117    6    1    0
    ##          C    0    1 1020    5    0
    ##          D    0    0   13  950    1
    ##          E    0    0    0    2 1080
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.9924          
    ##                  95% CI : (0.9898, 0.9944)
    ##     No Information Rate : 0.2868          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.9903          
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9911   0.9982   0.9817   0.9916   0.9991
    ## Specificity            0.9998   0.9954   0.9988   0.9972   0.9996
    ## Pos Pred Value         0.9994   0.9807   0.9942   0.9855   0.9982
    ## Neg Pred Value         0.9964   0.9996   0.9961   0.9984   0.9998
    ## Prevalence             0.2868   0.1901   0.1766   0.1628   0.1837
    ## Detection Rate         0.2843   0.1898   0.1733   0.1614   0.1835
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9954   0.9968   0.9902   0.9944   0.9993

Applying the Random Forest Model to Test Data
---------------------------------------------

The random forest model is applied to the 20 test cases available in the
test data provided.

    predict(fit_rf_ranger, val_data)

    ##  [1] B A B A A E D B A A B C B A E E A B B B
    ## Levels: A B C D E
