#########MULTICARE HEALTH SYSTEMS EMERGENCY REVISITS DATA########

## Read the data form the emergency revisit data file
dataset <- read.csv("Model_Input.csv")

dataset <- dataset[,c("Count_of_chronic_Diseases", "Derived_Admitting_Source", "Derived_Admitting_Status", "Facility", "Has_revisit_form", "Latest_Age", "Latest_County", "Latest_LOS", "Latest_Marital_Status", "Latest_Patient_Language", "Latest_Primary_Care_Named", "Latest_Race", "Latest_Religion", "Sex", "Year_of_Service")]

#dimension of the data filtered with age greater than or equal to 65 and payer type medicare
dim(dataset)

#Get the list of columns from the data
names(dataset)

###############Exploratory analysis######################

#To find the relation between age and the target variable
boxplot(Latest_Age ~ Has_revisit_form ,dataset)
#It is obseved that the median age for the visits and revisists is similar. Hence the varaible may
#not be significant when observed individually. Can be verified if the variable is significant when
#passed into the model with other variables.

#To find the relation between age and the target variable
boxplot(Latest_LOS ~ Has_revisit_form ,dataset)
#It is observed that most of the visits have LOS as 1, Hence transforming the varaible to categorical
#with levels as LOS =1 and LOS greater than 1

#Creating a new variable is_LOS_1 
dataset$is_LOS_1 <- as.factor(ifelse(dataset$Latest_LOS>1,"more than 1",1))

# Verify the transformation
table(dataset$is_LOS_1)

#Transforming the target variable as factor with two levels 0 and 1
dataset$Has_revisit_form <- as.factor(dataset$Has_revisit_form)


#Transforming the Year of service variable as factor with 4 levels 
dataset$Year_of_Service <- as.factor(dataset$Year_of_Service)

#Recoding the levels of county into three main groups- Pierce,King and other
dataset$Latest_County <- as.factor(ifelse((dataset$Latest_County == "Pierce County") |
                                   (dataset$Latest_County == "King County"),
                                   as.character(dataset$Latest_County),"Other" ))
# Verify the recoding
table(dataset$Latest_County)


################Partition the data#############################

#Set the seed
set.seed(1)

#Partition 60% of the data as train  data
train.rows <- sample(rownames(dataset), dim(dataset)[1]*0.6)

#Partition 30% of the data as Validation data
valid.rows <- sample(setdiff(rownames(dataset), train.rows),
                     dim(dataset)[1]*0.3)

# assign the remaining 10% as test data
test.rows <- setdiff(rownames(dataset), union(train.rows, valid.rows))

# create the train,validation and test data frames 
train.data <- dataset[train.rows, ]
valid.data <- dataset[valid.rows, ]
test.data <- dataset[test.rows, ]


###############oversampling the train data####################
# The dataset is an imbalanced data with more number of zeros than ones. Applying oversampling 
#will sample the minority class to get equal proportions of data for two classes.

library(ROSE)


#There are 41323 records with class - 0 . The minority class(1) is sampled to get 41323 records.
#total of 82646 records

data_balanced_over <- ovun.sample(Has_revisit_form ~ ., data = train.data, 
                                        method = "over",N = 82646)$data

#Verify the oversampling
table(data_balanced_over$Has_revisit_form)

#Selecting the input columns to the model.
inputcols <- c("Year_of_Service","Facility","Latest_Age","Has_revisit_form",
                             "Sex", "Latest_County","Latest_Marital_Status",
                             "Latest_Patient_Language" ,
                             "Latest_Race" ,"Latest_Religion","is_LOS_1",
                             "Latest_Primary_Care_Named", "Derived_Admitting_Source",
                             "Derived_Admitting_Status", 
                             "Count_of_chronic_Diseases")

#####################Modelling#############################################

#From Azure ML, we have observed that the best classifier for the MHS data
#is random forest.

library(randomForest)

## Develop model using random forest classifier using  the oversampled train data
rf <- randomForest(Has_revisit_form ~ ., data = data_balanced_over[,inputcols], ntree = 50,
                  mtry=10, nodesize = 1000, importance = TRUE)

## variable importance plot
varImpPlot(rf, type = 1)

## Probablilities of revisit on the validation data
rf.pred_prob <- predict(rf,valid.data,type="prob")[,2]

#to find the auc of the model on the validation data
library(ModelMetrics)
auc(valid.data$Has_revisit_form,rf.pred_prob)

## Predicted values of revisit on the valiadation data
rf.pred <- predict(rf,valid.data)

#confusion matrix
library(caret)
library(lattice)
library(ggplot2)
caret ::confusionMatrix(rf.pred, valid.data$Has_revisit_form)


## Predicted values of revisit on the complete dataset
rf.pred.comp <- predict(rf,dataset)

#confusion matrix
library(caret)
library(lattice)
library(ggplot2)
caret ::confusionMatrix(rf.pred.comp, dataset$Has_revisit_form)

##plot ROC courve
library(pROC)
r <- roc(valid.data$Has_revisit_form, rf.pred_prob)
plot.roc(r,col = "red",lwd = 3,auc.polygon =1, auc.polygon.col="light blue",
         identity.col="dark orange",identity.lwd=2, max.auc.polygon=1 )

##########Performance evaluation on train data to check overfitting#############

## Probablilities of revisit on the train data
rf.predtr_prob <- predict(rf, data_balanced_over,type="prob")[,2]

##AUC of the model on train data
library(ModelMetrics)
auc(data_balanced_over$Has_revisit_form,rf.predtr_prob)


## Predicted values of revisit on the train data
rf.predtr <- predict(rf,data_balanced_over)

library(caret)
caret :: confusionMatrix(rf.predtr, data_balanced_over$Has_revisit_form)

##sample decision tree
library(rpart)
library(rpart.plot)
default.ct <- rpart(Has_revisit_form ~ ., data =data_balanced_over, method = "class")
prp(default.ct, type =5, extra = 2, under = TRUE,split.font = .5, varlen = -30, faclen = 20)

###########################bagging#####################################

library(adabag)
bag <- bagging(Has_revisit_form ~ ., data = data_balanced_over)

#To predict the values
library(caret)
pred <- predict(bag, valid.data)
confusionMatrix(as.factor(pred$class), valid.data$Has_revisit_form)

#To predict the probabilities
bag.pred_prob <- predict(bag, valid.data,type="prob")
ModelMetrics::auc(valid.data$Has_revisit_form,bag.pred_prob$prob)

