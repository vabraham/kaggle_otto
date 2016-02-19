# Clear environment workspace
rm(list=ls())

# Install packages
install.packages("devtools")
require(devtools)
devtools::install_github('dmlc/xgboost',subdir='R-package')
install.packages("methods")
require(xgboost)
require(methods)

# Load data
train <- read.csv("/Users/vabraham24/Documents/RStudio/kaggle_otto/data/train.csv")
test <- read.csv("/Users/vabraham24/Documents/RStudio/kaggle_otto/data/test.csv")
samplesub <- read.csv("/Users/vabraham24/Documents/RStudio/kaggle_otto/data/sampleSubmission.csv")

# Remove id column so it doesn't get picked up by the current classifier
train <- train[,-1]
test = test[,-1]

# Setup data
y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass]

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 9,
              "nthread" = 8)

# Training parameters
param["eta"] <- 0.3   # Learning rate
param["max_depth"] <- 12  # Tree depth
nround = 100     # Number of trees to fit

# Train the model
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround, verbose=2)

# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)

# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='submission.csv', quote=FALSE,row.names=FALSE)
