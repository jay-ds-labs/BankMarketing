###################################################################################
# PART 0 - Setting up R environment
# PART 1 - Data Import, Audit & Cleaning
# PART 2 - Data Exploration
# PART 3 - Data Preparation for Model Building
# PART 4 - Decision Tree Based Model - Identifying cohorts with high probability to subscribe a term deposit
# PART 5 - Logistic Regression Model
# PART 6 - Random Forest Model
# PART 7 - GBM Model
# PART 8 - Ensemble Model
# PART 9 - Model finalization
# Part 10 - Finalizing test data percentage
###################################################################################

###################################################################################
# PART 0 - Setting up R environment
# 1. Installing packages if required
###################################################################################
# Clean global environment
rm(list=ls())
# Set working directory
# Please change this as per your choice
setwd(getwd())

# Required package list
required.packages.data.manipulation <- c('Hmisc','data.table','plyr','tidyverse','pander')
required.packages.visualization <- c('RColorBrewer','ggplot2','gridExtra')
required.packages.model <- c('car','caret','party','pROC','h2o')
required.packages.authoring <- c('rmarkdown','binb')
required.packages <- c(required.packages.data.manipulation,
                       required.packages.visualization,
                       required.packages.model,
                       required.packages.authoring)

# Installing required packages if needed
packages.to.install <- required.packages[which(!required.packages %in% installed.packages()[,1])]
if(length(packages.to.install)>0) {
  cat('Following packages will be installed:\n', packages.to.install)
  install.packages(packages.to.install)
  packages.to.install <- required.packages[which(!required.packages %in% installed.packages()[,1])]
}
if(length(packages.to.install)>0) cat('Failed to install:\n', packages.to.install) else print('All required packages are installed.')


###################################################################################
# PART 0 - Setting up R environment
# 2. Loading required packages & functions in memory
###################################################################################

sapply(required.packages, require, character.only = TRUE)
rm(required.packages.data.manipulation, required.packages.visualization, required.packages.model, 
   required.packages.authoring, required.packages, packages.to.install)

# Functions are created to make this code modular and easy to understand.
# Key tasks are done using functions kept in InternalFunctions.R file.
# All functions are required to run the code.
source('InternalFunctions.R')



###################################################################################
# PART 1 - Data Import, Audit & Cleaning
# 1. Importing bank marketing dataset from UCI Machine Learning Repository
###################################################################################

# Dataset download link: https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
# Dataset original source: Source: [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014

# data.load function will load the data from the UCI link
dt <- data.load('https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip')

# Performing basic audit on data loaded
data.audit(dt)

# Understanding:
# - Overall data has been loaded properly.
# - There are no missing values.
# - Data type of each variable is correct.
# - In the dependent variable 'y' we have 11.7% 'yes' in both training & validation dataset.
# - Outliers might be possible in variables - pdays, previous, campaign, duration, balance.
# - In variable 'contact' we see that 6.4% observations are 'telephone' while 28.8% is 'unknown' and rest is 'cellular'. It is possible that all the unknown are actually telephone. We will consider making only 1 dummy variable for 'cellular'.
# - All the character variables needs dummy variable (one-hot encoding) creation.
# - Variable 'day' although considered as integer, will need to be considered as factor.
# - Apart from 'day' variable transformation, no other data preparation is required. We will perform dummy creation post bi-variate analysis.

# Creates the data dictionary from the file - 'bank-names.txt'
show.data.dictionary()
###################################################################################


###################################################################################
# PART 2 - Data Exploration
# 1. Exploring categorical variables
###################################################################################

# univ.categ function performs uni-variate analysis on categorical data
univ.categ(dt, 'y', acc = 0.1) # 12% of customers have taken term deposit
univ.categ(dt, 'job') # Technician, management & blue-collar comprises of 60% of all job types
univ.categ(dt, 'marital') # Majority of customers are married
univ.categ(dt, 'education') # Majority of customers have completed secondary education. There are 4% unknown.
univ.categ(dt, 'default') # Mostly there is only 1 value. 2% of customers have credit in default
univ.categ(dt, 'housing') # A little more than half of all customers have taken a housing loan
univ.categ(dt, 'loan') # High majority of customers do not have a personal loan
univ.categ(dt, 'contact') # Majority of customers can be contacted using mobile. Unknowns are quite higher than telephone.
univ.categ(dt, 'day') 
univ.categ(dt, 'month') # Almost one third of customers are contacted in May
univ.categ(dt, 'poutcome') # Status of previous campaigns are unknown for 82% of cases. Only 3% have resulted in success earlier


###################################################################################
# PART 2 - Data Exploration
# 2. Exploring continuous variables
###################################################################################

# univ.cont function performs uni-variate analysis on continuous data
univ.cont(dt, 'age') # Higher outliers present. Outlier Cutoff 99.9%ile = 83 years will be considered later

univ.cont(dt, 'balance') # Lets perform log transformation as the range is very high and histogram is skewed
univ.cont(dt, 'balance', log.transform = T) # We will consider the log transformation. 
# Note that for log transformation, values = 0 are retained as 0 and for negative values -log10(abs(x)) is used
# We will create a derived variable with the following levels:
# 1: less than -2.5 , 2: between -2.5 to 0, 3: equal to 0, 4: between 0 and 2.5, 5: greater than 2.5

univ.cont(dt, 'duration') # Lets perform log transformation as the range is very high and histogram is skewed
univ.cont(dt, 'duration', log.transform = T) # We will consider log transformation and also outlier treatment. 
# Values of log transformation more than 3.3 or less than 0.7 will be considered as outliers

univ.cont(dt, 'campaign') # Majority of values is 0. We will consider this variable as is with higher outlier treatment
# Higher outlier cutoff will be 32

univ.cont(dt,'pdays') # Majority of values is 0. We will consider this variable as is with higher outlier treatment
# Higher outlier cutoff will be 637

univ.cont(dt,'previous') # Majority of values is 0. We will consider this variable as is with higher outlier treatment
# Higher outlier cutoff will be 22


###################################################################################
# PART 2 - Data Exploration
# 3. Exploring relationship of categorical variables with dependent variable y
# We will perform bi-variate analysis on categorical variables for appropriate dummy variable creation
###################################################################################

biv.categ(dt, 'job') # 4 dummy variables will be created for job type - student, retired, unemployed, technician or self-employed or admin or unknown or management
biv.categ(dt, 'marital') # 2 dummy variables will be created for marital status - single, divorced
biv.categ(dt, 'education') # 3 dummy variables will be created for education type - tertiary, unknown , secondary
biv.categ(dt, 'default') # 1 dummy variable will be created for default - no
biv.categ(dt, 'housing') # 1 dummy variable will be created for housing loan - no
biv.categ(dt, 'loan') # 1 dummy variable will be created for personal loan - no
biv.categ(dt, 'contact') # 2 dummy variables will be created for contact type - cellular, unknown
biv.categ(dt, 'day') # 5 dummy variables will be created for day - 1, 10, 30 or 3 or 22 or 25 or 4, 12 or 13 or 2 or 24 or 27 or 15 or 23 or 16, 11 or 9 or 26 or 5 or 14 or 8
biv.categ(dt, 'month') # 4 dummy variables will be created for month - mar, oct or dec or sep, feb, apr
biv.categ(dt, 'poutcome') # 3 dummy variables will be created for previous outcome - success, other, failure



###################################################################################
# PART 3 - Data Preparation for Model Building
# 1. Dummy variable creation
###################################################################################

# Dummy creation for - job
dt <- dt %>% mutate(d.job.1 = ifelse(job == 'student',1,0),
                    d.job.2 = ifelse(job == 'retired',1,0),
                    d.job.3 = ifelse(job == 'unemployed',1,0),
                    d.job.4 = ifelse(job %in% c('technician' , 'self-employed','admin.','unknown','management'),1,0))

# Check
# table(dt[,c('job','d.job.1')])
# table(dt[,c('job','d.job.2')])
# table(dt[,c('job','d.job.3')])
# table(dt[,c('job','d.job.4')])

# Dummy creation for - marital
dt <- dt %>% mutate(d.marital.1 = ifelse(marital == 'single',1,0),
                    d.marital.2 = ifelse(marital == 'divorced',1,0))
# Check
# table(dt[,c('marital','d.marital.1')])
# table(dt[,c('marital','d.marital.2')])

# Dummy creation for - education
dt <- dt %>% mutate(d.education.1 = ifelse(education == 'tertiary',1,0),
                    d.education.2 = ifelse(education == 'unknown',1,0),
                    d.education.3 = ifelse(education == 'secondary',1,0))
# Check
# table(dt[,c('education','d.education.1')])
# table(dt[,c('education','d.education.2')])
# table(dt[,c('education','d.education.3')])


# Dummy creation for - default
dt <- dt %>% mutate(d.default.1 = ifelse(default == 'no',1,0))
# Check
# table(dt[,c('default','d.default.1')])

# Dummy creation for - housing
dt <- dt %>% mutate(d.housing.1 = ifelse(housing == 'no',1,0))
# Check
# table(dt[,c('housing','d.housing.1')])


# Dummy creation for - loan
dt <- dt %>% mutate(d.loan.1 = ifelse(loan == 'no',1,0))
# Check
# table(dt[,c('loan','d.loan.1')])


# Dummy creation for - contact
dt <- dt %>% mutate(d.contact.1 = ifelse(contact == 'cellular',1,0),
                    d.contact.2 = ifelse(contact == 'unknown',1,0))
# Check
# table(dt[,c('contact','d.contact.1')])
# table(dt[,c('contact','d.contact.2')])


# Dummy creation for - day
dt <- dt %>% mutate(d.day.1 = ifelse(day == 1,1,0),
                    d.day.2 = ifelse(day == 10,1,0),
                    d.day.3 = ifelse(day %in% c(30,3,22,25,4),1,0),
                    d.day.4 = ifelse(day %in% c(12,13,2, 24,27,15,23,16),1,0),
                    d.day.5 = ifelse(day %in% c(11,9,26,5,14,8),1,0))
# Check
# table(dt[,c('day','d.day.1')])
# table(dt[,c('day','d.day.2')])
# table(dt[,c('day','d.day.3')])
# table(dt[,c('day','d.day.4')])
# table(dt[,c('day','d.day.5')])



# Dummy creation for - month
dt <- dt %>% mutate(d.month.1 = ifelse(month == 'mar',1,0),
                    d.month.2 = ifelse(month %in% c('oct','dec','sep'),1,0),
                    d.month.3 = ifelse(month == 'feb',1,0),
                    d.month.4 = ifelse(month == 'apr',1,0))
# Check
# table(dt[,c('month','d.month.1')])
# table(dt[,c('month','d.month.2')])
# table(dt[,c('month','d.month.3')])
# table(dt[,c('month','d.month.4')])


#success, other, failure
# Dummy creation for - poutcome
dt <- dt %>% mutate(d.poutcome.1 = ifelse(poutcome == 'success',1,0),
                    d.poutcome.2 = ifelse(poutcome == 'other',1,0),
                    d.poutcome.3 = ifelse(poutcome == 'failure',1,0))
# Check
# table(dt[,c('poutcome','d.poutcome.1')])
# table(dt[,c('poutcome','d.poutcome.2')])
# table(dt[,c('poutcome','d.poutcome.3')])


# Converting dependent variable to 0 & 1
dt$y <- ifelse(dt$y=='yes',1,0)

# Removing categorical variables whose dummy variables has been created
dt <- dt %>% select(-c(2:5,7:11,16))


###################################################################################
# PART 3 - Data Preparation for Model Building
# 2. Creating log transformation variables for continous variables with high range and skewed histogram
###################################################################################

# Log transformation for balance
dt$l.balance <- ifelse(dt$balance ==0, 0, ifelse(dt$balance<0, -log10(abs(dt$balance)), log10(dt$balance)))
# Check
# sum(is.nan(dt$l.balance))
# sum(is.na(dt$l.balance))
# sum(is.infinite(dt$l.balance))

# Log transformation for duration
dt$l.duration <- ifelse(dt$duration ==0, 0, ifelse(dt$duration<0, -log10(abs(dt$duration)), log10(dt$duration)))
# Check
# sum(is.nan(dt$l.duration))
# sum(is.na(dt$l.duration))
# sum(is.infinite(dt$l.duration))

# Removing the variables whose log transformation is done
dt <- dt[,-c(2,3)]



###################################################################################
# PART 3 - Data Preparation for Model Building
# 3. Creating a flag for all outliers
###################################################################################

dt$is.outlier <- F
dt <- dt %>% mutate(is.outlier = ifelse(age>=83,T,is.outlier)) # 63 outliers
dt <- dt %>% mutate(is.outlier = ifelse(l.duration>=3.3 | l.duration<=0.7,T,is.outlier)) # 120 outliers
dt <- dt %>% mutate(is.outlier = ifelse(campaign>=32,T,is.outlier)) # 47 outliers
dt <- dt %>% mutate(is.outlier = ifelse(pdays>=637,T,is.outlier)) # 41 outliers
dt <- dt %>% mutate(is.outlier = ifelse(previous>=22,T,is.outlier)) # 49 outliers

# Percentage of observations detected as outliers
100*prop.table(table(dt$is.outlier)) # 0.7% observations detected as outliers


###################################################################################
# PART 3 - Data Preparation for Model Building
# 4. Creating grouped variables
###################################################################################

# Creating grouped variable for log of balance
dt <- dt %>% mutate(g.l.balance = ifelse(l.balance>-2.5,ifelse(l.balance==0,3,ifelse(l.balance>0,ifelse(l.balance>2.5,5,4),2)),1))
# Check
# aggregate(l.balance~g.l.balance, data = dt,min)
# aggregate(l.balance~g.l.balance, data = dt,max)


###################################################################################
# PART 3 - Data Preparation for Model Building
# 5. Remove multicollinear variables using VIF test
###################################################################################
lm.out <- lm(y~.-g.l.balance, data = dt)
sort(vif(lm.out))

lm.out <- lm(y~.-g.l.balance -pdays, data = dt)
sort(vif(lm.out))

lm.out <- lm(y~.-g.l.balance -pdays-d.contact.2, data = dt)
sort(vif(lm.out))

lm.out <- lm(y~.-g.l.balance -pdays-d.contact.2-d.education.1, data = dt)
sort(vif(lm.out))

rm(lm.out)

# All VIF values are now less than 2
# Removing variables with high VIF. Creating a backup of dt, before we do this.
dt.backup <- dt
dt <- dt[, -c(3,19,12)] # We will keep g.l.balance. When building model we will always try only one between l.balance & g.l.balance


###################################################################################
# PART 3 - Data Preparation for Model Building
# 5. Derived variable creation with the help of CHAID
# We will build a decision tree to find cohorts where there is a higher probability to find y = 1
###################################################################################

# We want to find cohorts where y=1 for at least 12% of cases & minimum size of cohort is 500 
m.chaid3 <- ctree(y ~ ., data = dt, controls = ctree_control(testtype = "Univariate",maxdepth = 3))
plot.chaid(dt,m.chaid3,S = 500, P = 11.7, D= 3)

m.chaid4 <- ctree(y ~ ., data = dt, controls = ctree_control(testtype = "Univariate",maxdepth = 4))
plot.chaid(dt,m.chaid4,S = 500, P = 11.7, D= 4)

m.chaid5 <- ctree(y ~ ., data = dt, controls = ctree_control(testtype = "Univariate",maxdepth = 5))
plot.chaid(dt,m.chaid5,S = 500, P = 11.7, D= 5)

m.chaid6 <- ctree(y ~ ., data = dt, controls = ctree_control(testtype = "Univariate",maxdepth = 6))
plot.chaid(dt,m.chaid6,S = 500, P = 11.7, D= 6)

# We will go ahead with m.chaid3 and create 5 dummy variables as they cover maximum count of y =1 with highest proportion
rm(m.chaid4,m.chaid5,m.chaid6)

# Adding nodes to dt
dt$chaid.node <- predict(m.chaid3, newdata = dt, type="node")
# Creating dummy variables for node = 14, 8,5,15, 11
dt <- dt %>% mutate(node.14 = ifelse(chaid.node == 14, 1,0))
dt <- dt %>% mutate(node.8 = ifelse(chaid.node == 8, 1,0))
dt <- dt %>% mutate(node.5 = ifelse(chaid.node == 5, 1,0))
dt <- dt %>% mutate(node.15 = ifelse(chaid.node == 15, 1,0))
dt <- dt %>% mutate(node.11 = ifelse(chaid.node == 11, 1,0))
# Check
# table(dt$node.14)
# table(dt$node.8)
# table(dt$node.5)
# table(dt$node.15)
# table(dt$node.11)

# Dropping node variable
dt$chaid.node <- NULL


# We will need to check the VIF once again with the node dummy variables
lm.out <- lm(y~.-g.l.balance, data = dt)
sort(vif(lm.out))
# All variables have VIF less than 2
rm(lm.out)

###################################################################################
# PART 3 - Data Preparation for Model Building
# 6. Creating final datasets for model building

# We will create 6 sets of training & validation datasets -
# 1. Training dataset with 90% observations and outlier treatment
# 2. Training dataset with 90% observations without outlier treatment
# 3. Training dataset with 85% observations and outlier treatment
# 4. Training dataset with 85% observations without outlier treatment
# 5. Training dataset with 80% observations and outlier treatment
# 6. Training dataset with 80% observations without outlier treatment

# We will run several models on first set of training & validation i.e. Training dataset with 90% observations and outlier treatment
# The modeling technique selected from above basis F1 accuracy metric (on validation) will be applied on all other sets of training & validation 
# The best model will be then selected basis F1 accuracy metric on validation 
###################################################################################

train.n.validation <- data.split(dt, train.percentage = 0.9, outlier.treatment = T)
train90t <- train.n.validation$train
validation90t <- train.n.validation$validation

train.n.validation <- data.split(dt, train.percentage = 0.9, outlier.treatment = F)
train90 <- train.n.validation$train
validation90 <- train.n.validation$validation

train.n.validation <- data.split(dt, train.percentage = 0.85, outlier.treatment = T)
train85t <- train.n.validation$train
validation85t <- train.n.validation$validation

train.n.validation <- data.split(dt, train.percentage = 0.85, outlier.treatment = F)
train85 <- train.n.validation$train
validation85 <- train.n.validation$validation

train.n.validation <- data.split(dt, train.percentage = 0.8, outlier.treatment = T)
train80t <- train.n.validation$train
validation80t <- train.n.validation$validation

train.n.validation <- data.split(dt, train.percentage = 0.8, outlier.treatment = F)
train80 <- train.n.validation$train
validation80 <- train.n.validation$validation

rm(train.n.validation)


#save.image(file='image1.rdata')

###################################################################################
# PART 4 - Model Building
# 0. Creating a table to store model iteration results
###################################################################################
model.results <- data.frame(SNo = integer(), ModelType = character(), Model.Parameters = character(), Accuracy.train = double(), Accuracy.validation = double(), F1.train = double(), F1.validation = double(), stringsAsFactors=F)

###################################################################################
# PART 4 - Model Building
# 1. Logistic Regression model
# Models will be built on training dataset with 90% observation & outlier treatment done
###################################################################################

# Model iterations
m.log.reg <- glm(y~. -g.l.balance , data=train90t, family=binomial())
summary(m.log.reg)

m.log.reg <- glm(y~. -g.l.balance -d.default.1, data=train90t, family=binomial())
summary(m.log.reg)

m.log.reg <- glm(y~. -g.l.balance -d.default.1 -d.poutcome.3, data=train90t, family=binomial())
summary(m.log.reg)

m.log.reg <- glm(y~. -g.l.balance -d.default.1 -d.poutcome.3 -age, data=train90t, family=binomial())
summary(m.log.reg)

m.log.reg <- glm(y~. -g.l.balance -d.default.1 -d.poutcome.3 -age - d.education.2, data=train90t, family=binomial())
summary(m.log.reg)

m.log.reg <- glm(y~. -g.l.balance -d.default.1 -d.poutcome.3 -age - d.education.2 - d.job.3, data=train90t, family=binomial())
summary(m.log.reg)
# Above model has all significant variables

# Scoring on validation dataset
pred.log.reg <- get.predictions(m.log.reg, train90t, validation90t)

# Confusion matrix on validation data
confusion.matrix(validation90t$y, pred.log.reg)
# This model has Accuracy of 80.8% and F1 of 51.6%

# Lets try the above model by replacing l.balance with g.l.balance 
m.log.reg <- glm(y~. -l.balance -d.default.1 -d.poutcome.3 -age - d.education.2 - d.job.3, data=train90t, family=binomial())
summary(m.log.reg)
pred.log.reg <- get.predictions(m.log.reg, train90t, validation90t)
confusion.matrix(validation90t$y, pred.log.reg)
# This model has lower accuracy & f1, hence we revert back to the previous model 
m.log.reg <- glm(y~. -g.l.balance -d.default.1 -d.poutcome.3 -age - d.education.2 - d.job.3, data=train90t, family=binomial())

# Confusion matrix on training data
pred.log.reg.t <- get.predictions(m.log.reg, train90t, train90t)
cm.t <- confusion.matrix(train90t$y, pred.log.reg.t)
# Confusion matrix on validation data
pred.log.reg <- get.predictions(m.log.reg, train90t, validation90t)
cm.v <- confusion.matrix(validation90t$y, pred.log.reg)

model.results <- add.model.result(1,'Logistic Regression', 'NA', cm.t, cm.v, model.results)
view.model.results()


###################################################################################
# PART 4 - Model Building
# 2. Random Forest model
# Models will be built on training dataset with 90% observation & outlier treatment done
###################################################################################
# Initializing H2O instance
h2o.init(nthreads=-1, max_mem_size = "10G")  # initializes with all available threads and 10Gb memory
h2o.removeAll() # frees up the memory

train <- train90t
train$y <- as.factor(train$y)

##########     ITERATION 1      ###################
m.rf1 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=20, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
#summary(m.rf1)
# Performance on training data
pred.rf1.t <- h2o.predict(m.rf1, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf1.t$predict)$predict)
# Accuracy - 97.4% and F1 - 89.5%

# Performance on validation data
pred.rf1 <- h2o.predict(m.rf1, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf1$predict)$predict)
# We get Accuracy of 88.5% and F1 of 59.6%
# Performance of this model is better than our logistic regression model
# This is a clear case of overfitting
model.results <- add.model.result(2,'Random Forest', 'ntrees = 50, max_depth=20', cm.t, cm.v, model.results)
view.model.results()

##########     ITERATION 2      ###################
# Reducing the depth to reduce overfitting
m.rf2 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=15, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf2.t <- h2o.predict(m.rf2, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf2.t$predict)$predict)
# Accuracy - 93.8% and F1 - 75.7%

# Performance on validation data
pred.rf2 <- h2o.predict(m.rf2, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf2$predict)$predict)
# We get Accuracy of 88.9% and F1 of 60.1%
# Accuracy on train dropped but on validation it improved.
# There is still overfitting happening
model.results <- add.model.result(3,'Random Forest', 'ntrees = 50, max_depth=15', cm.t, cm.v, model.results)
view.model.results()



##########     ITERATION 3      ###################
# Reducing the depth further to reduce overfitting
m.rf3 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=10, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf3.t <- h2o.predict(m.rf3, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf3.t$predict)$predict)
# Accuracy - 90.3% and F1 - 64.1%

# Performance on validation data
pred.rf3 <- h2o.predict(m.rf3, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf3$predict)$predict)
# We get Accuracy of 88.9% and F1 of 59.9%
# Performance on training has dropped but on validation has remained almost the same
# Overfitting has been resolved though
model.results <- add.model.result(4,'Random Forest', 'ntrees = 50, max_depth=10', cm.t, cm.v, model.results)
view.model.results()


##########     ITERATION 4      ###################
# We will reduce the depth further, this time it might reduce the performance on validation,let's check
m.rf4 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf4.t <- h2o.predict(m.rf4, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf4.t$predict)$predict)
# Accuracy - 88.2% and F1 - 57.1%

# Performance on validation data
pred.rf4 <- h2o.predict(m.rf4, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf4$predict)$predict)
# We get Accuracy of 88.2% and F1 of 58.6%
# There is no overfitting now, but performance on validation has dropped
model.results <- add.model.result(5,'Random Forest', 'ntrees = 50, max_depth=5', cm.t, cm.v, model.results)
view.model.results()


##########     ITERATION 5      ###################
# We will retain the existing depth of 5 and increase the number of trees to 100 
m.rf5 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 100, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf5.t <- h2o.predict(m.rf5, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf5.t$predict)$predict)
# Accuracy - 88.7% and F1 - 57.7%

# Performance on validation data
pred.rf5 <- h2o.predict(m.rf5, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf5$predict)$predict)
# We get Accuracy of 88.6% and F1 of 58.8%
# There is no overfitting now, but performance on validation has dropped
model.results <- add.model.result(6,'Random Forest', 'ntrees = 100, max_depth=5', cm.t, cm.v, model.results)
view.model.results()

##########     ITERATION 6      ###################
# We will retain the existing depth of 5 and increase the number of trees to 100 
m.rf6 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 100, max_depth=10, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf6.t <- h2o.predict(m.rf6, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf6.t$predict)$predict)
# Accuracy - 90.2% and F1 - 63.9%

# Performance on validation data
pred.rf6 <- h2o.predict(m.rf6, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf6$predict)$predict)
# We get Accuracy of 88.7% and F1 of 59.9%
model.results <- add.model.result(7,'Random Forest', 'ntrees = 100, max_depth=10', cm.t, cm.v, model.results)
view.model.results()

##########     ITERATION 7      ###################
# We will retain the existing depth of 5 and increase the number of trees to 100 
m.rf7 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 200, max_depth=3, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf7.t <- h2o.predict(m.rf7, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf7.t$predict)$predict)
# Accuracy - 87.2% and F1 - 55.7%

# Performance on validation data
pred.rf7 <- h2o.predict(m.rf7, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf7$predict)$predict)
# We get Accuracy of 86.9% and F1 of 56.1%
model.results <- add.model.result(8,'Random Forest', 'ntrees = 200, max_depth=3', cm.t, cm.v, model.results)
view.model.results()


##########     ITERATION 8      ###################
# We will retain the existing depth of 5 and increase the number of trees to 100 
m.rf8 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 200, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf8.t <- h2o.predict(m.rf8, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf8.t$predict)$predict)
# Accuracy - 88.9% and F1 - 57.8%

# Performance on validation data
pred.rf8 <- h2o.predict(m.rf8, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf8$predict)$predict)
# We get Accuracy of 88.7% and F1 of 58.2%
model.results <- add.model.result(9,'Random Forest', 'ntrees = 200, max_depth=5', cm.t, cm.v, model.results)
view.model.results()

##########     ITERATION 9      ###################
m.rf9 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 100, max_depth=20, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf9.t <- h2o.predict(m.rf9, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf9.t$predict)$predict)

# Performance on validation data
pred.rf9 <- h2o.predict(m.rf9, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf9$predict)$predict)
# Accuracy - 88.7% & F1 - 59.9%
model.results <- add.model.result(10,'Random Forest', 'ntrees = 100, max_depth=20', cm.t, cm.v, model.results)
view.model.results()

##########     ITERATION 10      ###################
m.rf10 <- h2o.randomForest(y = "y", training_frame = as.h2o(train), ntrees = 500, max_depth=50, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random")        
# Performance on training data
pred.rf10.t <- h2o.predict(m.rf10, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.rf10.t$predict)$predict)

# Performance on validation data
pred.rf10 <- h2o.predict(m.rf10, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.rf10$predict)$predict)
# Accuracy - 88.7% & F1 - 60.1%
model.results <- add.model.result(11,'Random Forest', 'ntrees = 500, max_depth=50', cm.t, cm.v, model.results)
view.model.results()


##########    CONCLUSION OF RF MODEL     ###################
# We will choose m.rf2 with ntrees = 50, max_depth=15 as it gives the best Accuracy & F1 with lesser overfitting than similar performing iterations



###################################################################################
# PART 4 - Model Building
# 3. Gradient Boosting model
# Models will be built on training dataset with 90% observation & outlier treatment done
###################################################################################

##########     ITERATION 1      ###################
# We will start with similar model parameters as best random forest model
m.gbm1 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=15, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 
# Performance on training data
pred.gbm1.t <- h2o.predict(m.gbm1, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm1.t$predict)$predict)

# Performance on validation data
pred.gbm1 <- h2o.predict(m.gbm1, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm1$predict)$predict)

# Adding model results
model.results <- add.model.result(12,'GBM', 'ntrees = 50, max_depth=15', cm.t, cm.v, model.results)
view.model.results()
# For the similar model GBM has the least F1 and the highest Accuracy, that is because it predicts almost everything as 0 and very less as 1
# There is also a lot of overfitting


##########     ITERATION 2      ###################
# We will start with similar model parameters as best random forest model
m.gbm1 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=15, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 
# Performance on training data
pred.gbm1.t <- h2o.predict(m.gbm1, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm1.t$predict)$predict)

# Performance on validation data
pred.gbm1 <- h2o.predict(m.gbm1, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm1$predict)$predict)

# Adding model results
model.results <- add.model.result(12,'GBM', 'ntrees = 50, max_depth=15', cm.t, cm.v, model.results)
view.model.results()
# For the similar model GBM has the least F1 and the highest Accuracy, that is because it predicts almost everything as 0 and very less as 1
# There is also a lot of overfitting


##########     ITERATION 2      ###################
# Number to trees are reduced to reduce overfitting
m.gbm2 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=10, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm2.t <- h2o.predict(m.gbm2, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm2.t$predict)$predict)

# Performance on validation data
pred.gbm2 <- h2o.predict(m.gbm2, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm2$predict)$predict)

# Adding model results
model.results <- add.model.result(13,'GBM', 'ntrees = 50, max_depth=10', cm.t, cm.v, model.results)
view.model.results()
# This has improved the F1 and reduced overfitting


##########     ITERATION 3      ###################
# Reducing the number of trees further
m.gbm3 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm3.t <- h2o.predict(m.gbm3, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm3.t$predict)$predict)

# Performance on validation data
pred.gbm3 <- h2o.predict(m.gbm3, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm3$predict)$predict)

# Adding model results
model.results <- add.model.result(14,'GBM', 'ntrees = 50, max_depth=5', cm.t, cm.v, model.results)
view.model.results()
# This is the best performing iteration that we have seen till now
# There is also no overfitting


##########     ITERATION 4      ###################
# Lets increase the number of trees with same depth to see if it improves the performance
m.gbm4 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 100, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm4.t <- h2o.predict(m.gbm4, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm4.t$predict)$predict)

# Performance on validation data
pred.gbm4 <- h2o.predict(m.gbm4, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm4$predict)$predict)

# Adding model results
model.results <- add.model.result(15,'GBM', 'ntrees = 100, max_depth=5', cm.t, cm.v, model.results)
view.model.results()
# This has reduced the performance


##########     ITERATION 5      ###################
# We will go back to 50 trees and reduce depth
m.gbm5 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=4, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm5.t <- h2o.predict(m.gbm5, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm5.t$predict)$predict)

# Performance on validation data
pred.gbm5 <- h2o.predict(m.gbm5, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm5$predict)$predict)

# Adding model results
model.results <- add.model.result(16,'GBM', 'ntrees = 50, max_depth=4', cm.t, cm.v, model.results)
view.model.results()
# This has also dropped the performance


##########     ITERATION 6      ###################
# We will keep the same number of trees and increase the depth to 6
m.gbm6 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 50, max_depth=6, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm6.t <- h2o.predict(m.gbm6, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm6.t$predict)$predict)

# Performance on validation data
pred.gbm6 <- h2o.predict(m.gbm6, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm6$predict)$predict)

# Adding model results
model.results <- add.model.result(17,'GBM', 'ntrees = 50, max_depth=6', cm.t, cm.v, model.results)
view.model.results()
# This has not improved the performance



##########     ITERATION 7      ###################
# Going back to depth = 5 and decrease the trees
m.gbm7 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 45, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm7.t <- h2o.predict(m.gbm7, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm7.t$predict)$predict)

# Performance on validation data
pred.gbm7 <- h2o.predict(m.gbm7, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm7$predict)$predict)

# Adding model results
model.results <- add.model.result(18,'GBM', 'ntrees = 45, max_depth=5', cm.t, cm.v, model.results)
view.model.results()
# This has not improved the performance



##########     ITERATION 8      ###################
# We will keep depth = 5 and increase trees a little
m.gbm8 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 55, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm8.t <- h2o.predict(m.gbm8, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm8.t$predict)$predict)

# Performance on validation data
pred.gbm8 <- h2o.predict(m.gbm8, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm8$predict)$predict)

# Adding model results
model.results <- add.model.result(19,'GBM', 'ntrees = 55, max_depth=5', cm.t, cm.v, model.results)
view.model.results()
# This has improved the performance. we will continue to increase the trees slightly


##########     ITERATION 9      ###################
# We will keep depth = 5 and increase trees a little
m.gbm9 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 60, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm9.t <- h2o.predict(m.gbm9, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.gbm9.t$predict)$predict)

# Performance on validation data
pred.gbm9 <- h2o.predict(m.gbm9, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.gbm9$predict)$predict)

# Adding model results
model.results <- add.model.result(20,'GBM', 'ntrees = 60, max_depth=5', cm.t, cm.v, model.results)
view.model.results()
# This has improved the performance. we will continue to increase the trees slightly


##########    CONCLUSION OF GBM MODEL     ###################
# We will choose m.gbm8 with ntrees = 55, max_depth=5 as it gives the best Accuracy & F1 with least overfitting 



###################################################################################
# PART 4 - Model Building
# 4. Building the ensemble model with selected Random Forest & select Gradient Boosting model
# Models will be built on training dataset with 90% observation & outlier treatment done
###################################################################################

# ensemble model will be built with m.rf2 & m.gbm8
m.ensemble <- h2o.stackedEnsemble(y = "y",training_frame = as.h2o(train), 
                                  base_models = list(m.gbm8@model_id, m.rf2@model_id))

# Performance on training data
pred.ensemble.t <- h2o.predict(m.ensemble, as.h2o(train90t))
cm.t <- confusion.matrix(train90t$y, as.data.frame(pred.ensemble.t$predict)$predict)

# Performance on validation data
pred.ensemble <- h2o.predict(m.ensemble, as.h2o(validation90t))
cm.v <- confusion.matrix(validation90t$y, as.data.frame(pred.ensemble$predict)$predict)

# Adding model results
model.results <- add.model.result(21,'GBM + RF', 'NA', cm.t, cm.v, model.results)
view.model.results()
# We will reject the ensemble model as its performance is lower than the individual GBM model


###################################################################################
# PART 4 - Model Building
# 5. Trying the selected model (GBM with ntrees = 55, max_depth=5) on the following datasets -
# 90% Training without outlier treatment
# 85% Training with outlier treatment
# 85% Training without outlier treatment
# 80% Training with outlier treatment
# 80% Training without outlier treatment
###################################################################################


##########     ITERATION 1 -  90% Training without outlier treatment  ###################
train <- train90
train$y <- as.factor(train$y)

m.gbm11 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 55, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm11.t <- h2o.predict(m.gbm11, as.h2o(train90))
cm.t <- confusion.matrix(train90$y, as.data.frame(pred.gbm11.t$predict)$predict)

# Performance on validation data
pred.gbm11 <- h2o.predict(m.gbm11, as.h2o(validation90))
cm.v <- confusion.matrix(validation90$y, as.data.frame(pred.gbm11$predict)$predict)

# Adding model results
model.results <- add.model.result(22,'90% Train without outlier treatement', 'GBM: ntrees = 55, max_depth=5', cm.t, cm.v, model.results)
view.model.results()
# Model performance has reduced on dataset without outlier treatment, showing that outlier treatment is important
# We will hence, try the next iterations only on outlier treated data with 85% and 80% training 



##########     ITERATION 2 -  85% Training with outlier   ###################
train <- train85t
train$y <- as.factor(train$y)

m.gbm12 <- h2o.gbm(y = "y", training_frame = as.h2o(train), ntrees = 55, max_depth=5, nfolds = 3, seed = 1,keep_cross_validation_predictions = TRUE, fold_assignment = "Random") 

# Performance on training data
pred.gbm12.t <- h2o.predict(m.gbm12, as.h2o(train85t))
cm.t <- confusion.matrix(train85t$y, as.data.frame(pred.gbm12.t$predict)$predict)

# Performance on validation data
pred.gbm12 <- h2o.predict(m.gbm12, as.h2o(validation85t))
cm.v <- confusion.matrix(validation85t$y, as.data.frame(pred.gbm12$predict)$predict)

# Adding model results
model.results <- add.model.result(23,'85% Train with outlier treatment', 'GBM: ntrees = 55, max_depth=5', cm.t, cm.v, model.results)
view.model.results()
# Model performance with 85% training is lower than 90% training data
# It is fair to assume that the model performance will not improve with 80% training data.
# Hence we can select GBM with ntrees = 55, max_depth=5 on 90% Training Data with outlier treatment as the best model





###################################################################################
# PART 4 - Model Building
# 6. Lift Chart generation using final selected model on 90% training data with outlier treatment
###################################################################################

# m.gbm8 has given the best performance. This will be selected as the final model

# Lift chart on training data for best model
lift.chart.train <- generate.lift.chart.table(train90t$y, as.data.frame(pred.gbm8.t$p1)$p1)

# Lift chart on validation data for best model
lift.chart.validation <- generate.lift.chart.table(validation90t$y, as.data.frame(pred.gbm8$p1)$p1)

# Generating the lift chart to compare model performance on test & validation
ggplot(data = lift.chart.train, aes(x=Segment, y = CummPercentY_1)) + geom_point(color = 'blue') + geom_line(color = 'blue') +
  geom_line(aes(y=CummRandom), color = 'red') + 
  scale_x_continuous(name ="Deciles basis Predicted Probability to buy term deposit (Decreasing order)", breaks=seq(0,10,1))+
  labs(y = 'Cummulative percentage of buyers (y=1') + ggtitle('Lift Chart on Training & Validation Data', 'Model Performance Vs Random Selection')+
  geom_text(aes(x = 5, y=41, label = 'Performance in top 3 deciles'), size = 5, hjust = 0, color = 'black') +
  geom_text(aes(x = 5, y=34, label = 'Training - 92% of responders captured'), size = 4, hjust = 0, color = 'blue') +
  geom_text(aes(x = 5, y=27, label = 'Validation - 89% of responders captured'), size = 4, hjust = 0, color = 'darkgreen') +
  geom_text(aes(x = 5, y=20, label = 'Random - 30% of responders captured'), size = 4, hjust = 0, color = 'red') +
  geom_line(data = data.frame(x = c(3,3), y = c(0,92)), aes(x = x, y = y), linetype = "dashed") +
  geom_line(data = data.frame(x = c(0,3), y = c(92,92)), aes(x = x, y = y), linetype = "dashed") +
  geom_line(data = data.frame(x = c(0,3), y = c(30,30)), aes(x = x, y = y), linetype = "dashed") +
  geom_line(data = lift.chart.validation, aes(x=Segment, y = CummPercentY_1), color = 'darkgreen')+
  geom_line(data = data.frame(x = c(0,3), y = c(89,89)), aes(x = x, y = y), linetype = "dashed")

# key vars
summary(m.gbm8)
h2o.varimp(m.gbm8)
h2o.varimp_plot(m.gbm8,num_of_features =10)


