
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Analysis on the weight lifting execution with deep learning models

In this study, data collected using movement-tracking devices such as *Jawbone Up*, *Nike FuelBand*, and *Fitbit* were analized ([data source](http://groupware.les.inf.puc-rio.br/har)). The objective is to predict the exceution of weight lifting excersises (i.e.correct or incorrect) using Machine learning tecniques. First the data was pre-processed and dimensianality reduced, then to perfrom cross-validation a training a test sets where generated. Finally a Random Forest was executed and tested with external data.

##Data loading

```{r} 
trainingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(trainingURL, "./training.csv", method = "curl")
download.file(testingURL, "./testing.csv", method = "curl")

training <- read.csv("./training.csv", header = TRUE)
#testing is only used in the final part
testing <- read.csv("./testing.csv", header = TRUE)
```

##Data pre-processing
A large dataset on human activity was first read in broadly analyzed. First, the first 7 and the last (160) rows were discarted, because they refer to variables such as user name and adquisition time. The last (160) records the incorrect and correct performance on weight lifting classes and was just temporaly deleted for convinience. Then the classes of every variable were explored. Many numeric predictors were **miscategorized** as factors and some of them presented **less than 10 levels** (10 observations). These were discarted and the rest were transformed to **numeric** and **integer** values. 

```{r warning=FALSE}
##Removing unnecessary variables
trainingSS <- training[,-c(1:7,160)]
##Removing empty 'factor' variables
factor_levels<- sapply(trainingSS, function(x) length(levels(x)))
trainingSS <- trainingSS[,which(factor_levels>10 | factor_levels==0, arr.ind = T)] 
##Transforming all variables to 'numeric' and 'integer'
classes<-sapply(trainingSS,class)
trainingSS[,which(classes=="factor")]<-lapply(trainingSS[,which(classes=="factor")],function(x){as.numeric(levels(x))[x]})
table(sapply(trainingSS,class))
```

Then also some variables were discarted based on their variability and correlation with other variables. Thereby, first *nearZerovar* function was used for identifying these **low variablity** variables that should not be used in the prediction algorithm, and similarly also variables which presented **strong correlation** (>0.9) where discarted.     
```{r warning=FALSE}
library(caret)
#Removing low variability variables
nsv<- nearZeroVar(trainingSS,saveMetrics = T)
trainingSS <- trainingSS[,!nsv$nzv]
#Removing highly correlated variables
corr <- findCorrelation(cor(trainingSS,use="complete.obs"), cutoff = 0.90)
trainingSS <- trainingSS[,-c(corr)]
trainingSS <- cbind(classe=training$classe,trainingSS)
```

It was observed that even applying these pre-processing methods the dimensionality of the dataset reduced largely, however still some variables present large proportions of NA values. NA values were present in almost **30%** of the predictors, and at least in **97%** of their observations. So the next step is to discart them. However, the 3% of observations corresponds to still around 580 observation, which may have a relation to the outcome. Thus, also one training sample with NA values (training_na) will be evaluated in parallel.

```{r}
nas<- sapply(trainingSS, function(x) sum(is.na(x)))
barplot(nas,main="NA values", horiz=T,xlab="number of NA's",ylab="predictor",yaxt='n', ann=F)
##Creating a new dataset with NA values for evaluation
training_na<- trainingSS
##Removing variables with NA's
trainingSS<- trainingSS[,which(nas<19000, arr.ind = T)]

```

##Model fitting
The training data was then divided in two groups, one for **training** the model and the other for **validating** it with independent observations, the division corresponds to **70%** and **30%** of the entire training set, respectively.

```{r}
set.seed(300)
subSet <- createDataPartition(y = trainingSS$classe, p = 0.7,list = FALSE)
#Building training and validation subsets
trainSS <- trainingSS[subSet,]
validSS <- trainingSS[-subSet,]
#Also for the dataset with NA's
train_na<- training_na[subSet,]
valid_na<- training_na[-subSet,]
#Removing variables to free RAM space
rm(trainingSS,training_na,training,nsv)
```

In this step the training subsample is used to train a model using *Random Forest* method. This method is very usefull for dealing with large number of features and avoids overfitting, therefore predictions are very accurate. One dissadvantage however, is the processing time therefore this aspect is also meassured.

```{r warning=FALSE}
library(randomForest)
start_time <- Sys.time()
rfMod <- randomForest(classe ~ .,data = trainSS,importance=T)
end_time <- Sys.time()
end_time - start_time
```

```{r}
start_time <- Sys.time()
rfMod2 <- randomForest(classe ~ .,data = train_na,na.action = na.omit)
end_time <- Sys.time()
end_time - start_time
```

The processing time when only omitting NA values was much lower than when using the dataset without predictors with NA's.

##Cross-validation
Finally the resulted model was used to generate predictions using the validation set for both cases.
```{r}
validpred <- predict(rfMod, validSS)
confusionMatrix(validpred, validSS$classe)$overall
```
```{r}
validpred <- predict(rfMod2, valid_na)
confusionMatrix(validpred, valid_na$classe)$overall
```
```{r}
par(mfrow=c(1,2))
plot(rfMod, main="Model 1")
plot(rfMod2, main="Model 2")
```
The higher accuracy was obtained using only predictors **without NA predictors**. Omitting NA values, but still using the remaning observations resulted less accurate. Model 1 could even be tuned since in the figure, it can be seen that only **200 trees** were needed to reach a **minimum error**.  

```{r}
varImpPlot(rfMod,sort=T,n.var=10,main="top 10 Variables")
```

The variable that increase the model accuracy more and also reduced Gini, was **Magnet_dumbbelt_z**.

##Test set prediction
Finally the model is tested using the external data as it is.
```{r}
predtest<-predict(rfMod,testing)
print(predtest)
```

