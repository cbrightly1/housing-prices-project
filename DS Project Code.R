#
#
# STAT 1361 Final Project R Script
# By Clare Cruz
#
#
################# Data Set Up

  #library load
  library(plyr)
  library(corrplot)
  library(leaps)
  library(glmnet)
  library(caret)
  library(class)
  library(FNN)
  library(splines)
  library(tree)
  library(randomForest)
  library(gbm)
  library(gam)
  library(dplyr)
  library(pls)
  library(tidyr)
  
  #read in the data
  test.pred <- read.csv("C:\\Users\\cbrig\\OneDrive - University of Pittsburgh\\Inter Data Science\\Final Project\\test.csv")
  df <- read.csv("C:\\Users\\cbrig\\OneDrive - University of Pittsburgh\\Inter Data Science\\Final Project\\train-1.csv")

################ Data Exploration

  #str(df)
  
  # Separating the quant and qual variables into lists
  quant.list <- c("price","AvgIncome","lotarea","yearbuilt","fireplaces","totalrooms","numstories","bedrooms","bathrooms","sqft")
  qual.list <- c("desc","exteriorfinish","rooftype","state", "zipcode","basement")
  
  # Variable summaries
  # summary(df[,names(df) %in% quant.list])
  # sapply(df[,names(df) %in% qual.list],table)
  
  # Visual distribution of the quantitative variables
  # par(mfrow = c(2,3))
  # boxplot(df[,names(df) %in% c("totalrooms","numstories","bedrooms","bathrooms")])
  # boxplot(df[,names(df) %in% c("price")], xlab = "price")
  # boxplot(df[,names(df) %in% c("AvgIncome")], xlab = "AvgIncome")
  # boxplot(df[,names(df) %in% c("lotarea")], xlab = "lotarea")
  # boxplot(df[,names(df) %in% c("yearbuilt")], xlab = "yearbuilt")
  
  # Price and lot area are highly skewed, need to check price for potential outliers
  # Only one mobile home which could potentially cause issues
  # Some zipcodes only hand a handful of properties while others have a bunch
  # Only condimiums have 0 lot area
  
  # If we take the na values from fireplace out we only get properties from PA, and since we can't update 
  # the data and the state is important for out analysis we should get rid of fireplaces.
  # I'll get rid of this for now but you could create two separate models for VA and PA 
 
  # df[is.na(df$fireplaces),]
  # df.fire <- df[!is.na(df$fireplaces),]
  # table(df.fire[,names(df.fire) %in% "state"])
  
  # There's a strong positive relationship with price and total rooms, bedrooms, and bathrooms
  # There's signs of multicollinearity between bedrooms, bathrooms, sqft, and bedrooms which aligns in the context
  cor <- cor(df[,names(df) %in% quant.list])

############### Outliers and Influential Points

  # Redefining our data set to exclude fireplaces
  df.clean <- df[,-12]
  
  # Fitting full model for outlier testing
  df.lm <- lm(price~., data = df[,-c(1,12)])
  
  # Looks like there are a lot of outliers and influential points
  # plot(df.lm, which = c(1,2,4))
  
  # Cooks distance
  # No observations have particularly high cooks distance
  n=dim(df)[1]
  cooks.D<-cooks.distance(df.lm)
  # head(sort(cooks.D[cooks.D>4/n],decreasing = TRUE))
  
  # Studentized residuals
  # Obs 639, 1323, 32, 756, 1372, 1285, and 75 have particularly high residuals (all above 4)
  n=dim(df)[1];p=14
  t0<-qt(0.975,n-p-1)
  rstudent<-rstudent(df.lm)
  # head(sort(rstudent[abs(rstudent)>t0], decreasing = TRUE),15)
  
  # Leverage values
  # Obs 1195 and 569 have particularly high residuals (above 0.5)
  n=dim(df)[1];p=14
  lev0<-2*(p+1)/n
  lev<-hatvalues(df.lm)
  # head(sort(lev[lev>lev0], decreasing = TRUE),10)
  
  # After getting rid of the observations, there are less outliers/influential points but the data is definitely not normally distributed
  # Can't exclude 1195 because it's the only mobile home property
  df.clean <- df.clean[-c(639,569,1323,32,756,1372,1285,75),]
  lm.clean <- lm(price~., data = df.clean[,-1])
  # plot(lm.clean, which = c(1,2,4))

################# Normality Assumptions
  # Multicollinearity
  # State and zipcode have a high VIF but this makes sense because they both relate to location
  car::vif(lm.clean)

  # Since our data is skewed to the right, we will do a log transformation
  # The data is not normal by shapiro wilk, but histogram with trans shows that it's a lot better 
  df.log <- log(df.clean$price)
  # shapiro.test(df.log)
  par(mfrow = c(1,2))
  hist(df$price)
  hist(df.log)
  
  # Redfining the data set to log(price)
  df.clean$price <- log(df.clean$price)

################# Test and Training Sets
  set.seed(14)
  
  #Change char variables to factor
  cols <- c("rooftype","exteriorfinish","desc","state", "zipcode")
  df.clean[cols] <- lapply(df.clean[cols], factor)

  # Create training and test sets
  vec <- 1:dim(df.clean)[1]
  train <- sort(sample(nrow(df.clean), nrow(df.clean)*.7))
  test <- vec[-train] 
  df.train <- df.clean[train,]
  df.test <- df.clean[test,]

  # Breakdown needed for some models
  y.train <- df.train[,2]
  y.test <- df.test[,2]
  x.train <- df.train[,-c(1,2)]
  x.test <- df.test[,-c(1,2)]
  test.mat <- model.matrix(price~., data=df.test[,-1])
  train.mat <- model.matrix(price~., data=df.train[,-1])

  # Model list to keep track of test MSEs & models
  model.mses <- rep(NA,15)

################ Model Testing
### These are the models that will be tested and compared
### They are tested in the order that they appear below
###
### Linear Models
### 1.) Forward Selection
### 2.) Backward Selection
### 3.) Best Subset Selection
###
### Shrinkage, Regularization, & KNN
### 4.) Lasso
### 5.) Ridge Regression
### 6.) KNN
### 7.) PCA
### 8.) PLS
###  
### Nonlinear Models
### 9.) Polynomials
### 10.) Splines
### 11.) GAMs
###
### Trees and Esembles
### 
### 12.) Regression Tree
### 13.) Random Forest
### 14.) Bagging with Random Forest
### 15.) Boosted Random Forest

################# Forward Selection

  forward.mod <- regsubsets(price ~., data = df.train[,-c(1,15)], nvmax = 14, method = "forward")

  forward.mse <- NULL
  # Creates column names that follow the form "x.1"
  # do.NULL creates the new names and it defaults to counting the dim
  # Need col names to match the subset reg model names
  x_cols <- colnames(x.train, do.NULL = FALSE)
  
  for(i in 1:14){
    #Grab the model coefficients to make model
    coef <- coef(forward.mod, id=i)
    #1. Get the test set according to the variables in the model
    #2. Then multiply them by the model coefficients like y = B1X + B2X etc
    pred <- as.matrix(x.test[, x_cols %in% names(coef)]) %*% coef[names(coef) %in% x_cols]
    forward.mse <- c(forward.mse, mean((y.test-pred)^2))
  }
  
  # plot(c(1:14),forward.mse, type = "b", ylab = "Test MSE", xlab = "Model Index")
  # coef(forward.mod,which.min(forward.mse))
  model.mses[1] <- min(forward.mse)

################# Backward Selection

  back.mod <- regsubsets(price ~., data = df.train[,-c(1,15)], nvmax = 14, method = "back")

  back.mse <- NULL
   # Creates column names that follow the form "x.1"
   # do.NULL creates the new names and it defaults to counting the dim
   # Need col names to match the subset reg model names
  x_cols <- colnames(x.train, do.NULL = FALSE)
  
  for(i in 1:14){
    # Grab the model coefficients to make model
    coef <- coef(back.mod, id=i)
    # 1. Get the test set according to the variables in the model
    # 2. Then multiply them by the model coefficients like y = B1X + B2X etc
    pred <- as.matrix(x.test[, x_cols %in% names(coef)]) %*% coef[names(coef) %in% x_cols]
    back.mse <- c(back.mse, mean((y.test-pred)^2))
  }
  
  # plot(c(1:14),back.mse, type = "b", ylab = "Test MSE", xlab = "Model Index")
  # coef(back.mod,which.min(back.mse))
  model.mses[2] <- min(back.mse)

################# Best Subset Selection

  subset.mod <- regsubsets(price ~. , data = df.train[,-c(1,15)], nvmax = 14, really.big = TRUE)

  sub.mse <- NULL
  
  # Creates column names that follow the form "x.1"
  # do.NULL creates the new names and it defaults to counting the dim
  # Need col names to match the subset reg model names
  x_cols <- colnames(x.train, do.NULL = FALSE)
  
  for(i in 1:14){
    # Grab the model coefficients to make model
    coef <- coef(subset.mod, id=i)
    # 1. Get the test set according to the variables in the model
    # 2. Then multiply them by the model coefficients like y = B1X + B2X etc
    pred <- as.matrix(x.test[, x_cols %in% names(coef)]) %*% coef[names(coef) %in% x_cols]
    sub.mse <- c(sub.mse, mean((y.test-pred)^2))
  }
  
  # plot(c(1:14),sub.mse, type = "b", ylab = "Test MSE", xlab = "Model Index")
  # coef(subset.mod,which.min(sub.mse))
  model.mses[3] <- min(sub.mse)

################# Lasso
  
  # cv.glmnet does k-fold cross validation with any glmnet model, default is k = 10
  # alpha = 1 turns the model into lasso
  lasso.mod <- cv.glmnet(train.mat, df.train$price, alpha=1)
  best.lambda <- lasso.mod$lambda.min
  
  # Test error on the model with the best lambda
  lasso.mse <- mean((y.test - predict(lasso.mod, newx = test.mat, s =best.lambda))^2);lasso.mse
  # coef(lasso.mod)
  model.mses[4] <- lasso.mse
  
################# Ridge Regression
 
  # cv.glmnet does k-fold cross validation with any glmnet model, default is k = 10
  # alpha = 0 turns the model into ridge
  ridge.mod <- cv.glmnet(train.mat, df.train$price, alpha=0)
  best.lambda <- ridge.mod$lambda.min
  
  # Test error on the model with the best lambda
  ridge.mse <- mean((y.test - predict(ridge.mod, newx = test.mat, s =best.lambda))^2);ridge.mse
  # coef(ridge.mod)
  model.mses[5] <- ridge.mse

################# KNN
  
  # Performs LOOCV
  knn.fit <- knn.reg(train = train.mat,y = y.train)
  
  knn.mod <- knn.reg(train = train.mat,test = test.mat,y = y.train,k=knn.fit$k)
  knn.mse <- mean((y.test - knn.mod$pred)^2)
  # knn.mod$k
  model.mses[6] <- knn.mse
  
################# PCR
  
  pcr.mod <- pcr(price~., data = df.train[,-c(1,3,15)], scale = TRUE, validation = "CV")
  # summary(pcr.mod)
  # validationplot(pcr.mod ,val.type="MSEP")
  pcr.error <- mean((df.test$price-predict(pcr.mod,df.test,ncomp =2))^2)
  # coef(pcr.mod)
  model.mses[7] <- pcr.error

################# PLS
  
  pls.mod <- plsr(price~., data = df.train[,-c(1,3,15)], scale = TRUE, validation = "CV")
  # summary(pcr.mod)
  # validationplot(pls.mod ,val.type="MSEP")
  pls.error <- mean((df.test$price-predict(pls.mod,df.test,ncomp =2))^2)
  # coef(pls.mod)
  model.mses[8] <- pls.error
  
  
################# Splines
  spline.mse <- NULL
  
  # For each quant variable see which model has the smallest MSE
  # Three knots makes it a cubic spline
  for(j in c(5,9,10,11,12,16)){
    spline.mod <- glm(price~bs(df.train[,j], df = 3), data = df.train[,-1])
    spline.pred <- predict(spline.mod,df.test)
    spline.mse <- c(spline.mse,mean((df.test$price-spline.pred)^2))}
  
  # colnames(df.train)[c(5,9,10,11,12,16)[which.min(spline.mse)]]
  model.mses[9] <- min(spline.mse)
  
################# Polynomials
  
  # For each quant variable see which order polynomial has the smallest MSE
  # then out of every variable, which model has the smallest MSE
  var.mse <- NULL
  for(j in c(5,9,10,11,12,16)){
    poly.mse <- NULL
    for(i in 1:4){
      poly.mod <- lm(price~poly(df.train[,j],i), data = df.train[,-c(1,3)])
      poly.pred <- predict(poly.mod,df.test)
      poly.mse <- c(poly.mse,mean((df.test$price-poly.pred)^2))}
    var.mse <- c(var.mse,min(poly.mse))}
 
  # colnames(df.train)[c(5,9,10,11,12,16)[which.min(var.mse)]]
  model.mses[10] <- min(var.mse)
  
################# GAM
  
  gam.mod <- gam(price~desc+s(numstories) + s(yearbuilt) + exteriorfinish + rooftype + s(bedrooms) 
             + s(bathrooms) + s(sqft) + state + s(AvgIncome), data = df.train)
  gam.pred <- predict(gam.mod, df.test)
  gam.mse <- mean((df.test$price-gam.pred)^2)  
  # coef(gam.mod)
  model.mses[11] <- gam.mse
  
################# Regression Tree
  
  tree.mod <- tree(price~., data = df.train[,-c(1,15)])
  
  # The optimal tree complexity is the full tree
  cv.tree <- cv.tree(tree.mod)
  # plot(cv.tree$size, cv.tree$dev, type = 'b')
  
  # Visualizing the tree
  # plot(tree.mod)
  # text(tree.mod, pretty = 0)
  
  tree.pred <- predict(tree.mod, df.test)
  tree.mse <- mean((df.test$price - tree.pred)^2)
  # summary(tree.mod)
  model.mses[12] <- tree.mse
  
  
################# Random Forest
  
  rf.mod <- randomForest(price~., data = df.train[,-1], mtry = 14/3, importance = TRUE)
  rf.pred <- predict(rf.mod, df.test)
  rf.mse <- mean((df.test$price - rf.pred)^2)
  #importance(rf.mod)
  model.mses[13] <- rf.mse
  
################# Bagging Forest

  #mtry = p reduces to bagging
  bag.mod <- randomForest(price~., data = df.train[,-1], mtry = 14)
  bag.pred <- predict(bag.mod, df.test)
  bag.mse <- mean((df.test$price - bag.pred)^2)
  #importance(bag.mod)
  model.mses[14] <- bag.mse
  
################# Boosted Forest
  
  MSE.test <- NULL
  lambda <- seq(0,0.8, by = 0.05)
  
  for(i in lambda){
    boost.mod <- gbm(price~., data = df.train[,-1], distribution = "gaussian", n.trees = 1000, shrinkage = i)
    boost.pred <- predict(boost.mod, df.test, n.trees = 1000)
    MSE.test <- c(MSE.test,mean((df.test$price-boost.pred)^2))}
    # plot(lambda,y = MSE.test, type = "b", xlab = "Lambda/Shrinkage Value")
  
  #boost.mod$var.names
  model.mses[15] <- min(MSE.test)  

  
################## Analysis Results
  # The random forest based models did the best out of all the models
  # A simple random forest had the minimum MSE
  final.results <- data.frame(model.mses, c("FS","BS","BSS","Lasso", "Ridge","KNN",
                    "PCR","PLS","Poly","Spline","GAM","1 Tree","RF","BaggedRF","BoostedRF"))
  colnames(final.results) <- c("MSE","Model")
  final.results
  
  # Variable Importance for the best model
  # varImpPlot(rf.mod)
  
  # Variable Analysis
  zipcode.analysis <- df.clean %>%
    group_by(zipcode)%>%
    summarize(mean(price))
  
  par(mfrow = c(2,2))
  plot(df.clean$yearbuilt,exp(df.clean$price))
  plot(df.clean$sqft,exp(df.clean$price))
  plot(df.clean$bathrooms,exp(df.clean$price))
  plot(df.clean$lotarea,exp(df.clean$price))
  
  zipcode.analysis2 <- df.clean %>%
    group_by(zipcode)%>%
    summarize(mean(price),
              mean(lotarea))
  
  # Apply data transformations to new test set
  test.pred <- test.pred[,-c(12)]
  cols <- c("rooftype","exteriorfinish","desc","state","zipcode")
  test.pred[cols] <- lapply(test.pred[cols], factor)
  test.pred$price <- as.numeric(test.pred$price)
  levels(test.pred$zipcode) <- union(levels(test.pred$zipcode),levels(df.train$zipcode))
  
  # Get prediction, put it into a table and export the data
  final.pred <- predict(rf.mod,test.pred)
  final.pred <- exp(final.pred)
  pred.export <- data.frame(test.pred[,1],final.pred, rep(4265886,600))
  colnames(pred.export) <- c("id","price","student_id")
  write.csv(pred.export,"C:\\Users\\cbrig\\OneDrive - University of Pittsburgh\\Inter Data Science\\Final Project\\testing_predictions_4265886.csv")
  