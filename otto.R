# clear environment workspace
rm(list=ls())

# load data
train <- read.csv("~/Documents/RStudio/kaggle_otto/data/train.csv")
test <- read.csv("~/Documents/RStudio/kaggle_otto/data/test.csv")
sample_sub <- read.csv("~/Documents/RStudio/kaggle_otto/data/sampleSubmission.csv")

# remove id column so it doesn't get picked up by the random forest classifier
train2 <- train[,-1]

# install randomForest package
install.packages('randomForest')
library(randomForest)

# set a unique seed number so you get the same results everytime you run the below model,
# the number does not matter
set.seed(12)

# create a random forest model using the target field as the response and all 93 features as inputs
fit <- randomForest(as.factor(target) ~ ., data=train2, importance=TRUE, ntree=100)
varImpPlot(fit)

# use the random forest model to create a prediction
pred <- predict(fit,test,type="prob")
submit <- data.frame(id = test$id, pred)
write.csv(submit, file = "firstsubmit.csv", row.names = FALSE)



