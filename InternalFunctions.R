###################################################################################
# PART 1 - Data Import, Audit & Cleaning
# 1. Importing 10Mn Dataset & Creating Train & Test
###################################################################################

data.load <- function(download.link){
  dl <- file.path(getwd(),'dataset.zip')
  download.file(download.link, dl)
  data.all <- read.csv2(unzip(dl, "bank-full.csv"),stringsAsFactors = F)
  return(data.all)
}

# Function for performing basic data audit after creating test & train datasets
data.audit <- function(ds){
  options(scipen = 20)
  invisible(readline(prompt="Increase the width of your console. Press [enter] to continue"))
  cat('Top 10 rows of the dataset -')
  pander(head(ds,10), style='simple', split.table = 160)
  cat(rep('=',65),'\n\n\n')
  invisible(readline(prompt="Press [enter] to continue"))
  cat('Dataset structure -\n')
  glimpse(ds)
  cat(rep('=',65),'\n\n\n')
  invisible(readline(prompt="Press [enter] to continue"))
  cat('Dataset columnwise summary -\n')
  Hmisc::describe(ds)
}

show.data.dictionary <- function(){
  dl <- file.path(getwd(),'dataset.zip')
  data.dictionary <- as.data.frame(readLines(unzip(dl, "bank-names.txt"))[c(47:55,57:60,62:65,68)])
  data.dictionary[2,1] <- paste0(data.dictionary[2,1] ,trimws(data.dictionary[3,1]))
  data.dictionary <- as.data.frame(data.dictionary[-3,])
  data.dictionary[1,1] <- '   1 - age : (numeric)'
  names(data.dictionary) <- 'line'
  data.dictionary <- data.dictionary %>% separate(line, c("number", "detail"), sep = "-", extra = "merge") 
  data.dictionary$detail <- str_replace(data.dictionary$detail ,"-",":")
  data.dictionary <- data.dictionary %>% separate(detail, c("Variable", "Description"), sep = ":", extra = "merge") 
  data.dictionary$Description <-  trimws(data.dictionary$Description)
  data.dictionary$Description <- sub("^\\s+","",data.dictionary$Description)
  data.dictionary$Description <- gsub('"','',data.dictionary$Description)
  data.dictionary$Description <- substr(data.dictionary$Description, 1, 100)
  data.dictionary$Variable <- substr(data.dictionary$Variable, 1, 10)
  pander(data.dictionary, split.cell = 40, split.table = Inf)
}

###################################################################################
# PART 2 - Data Exploration
###################################################################################

# Performs uni-variate analysis on categorical data
univ.categ <- function(ds,var,sort=T, acc = 1){
  ds <- as.data.frame(prop.table(table(ds[,var])))
  if(sort == T){
    ds %>% ggplot(aes(x=reorder(Var1,Freq), y = Freq)) + geom_col(fill = "#FF7F24") + 
      geom_text(aes(label = scales::percent(Freq,accuracy=acc)), vjust = -0.5, size = 3) +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
      scale_y_continuous(labels = scales::percent) + labs(y = 'Percentage of observations', x = var) + 
      ggtitle(paste('Distribution of',var))
  }else{
    ds %>% ggplot(aes(x=Var1, y = Freq)) + geom_col(fill = "#FF7F24") + 
      geom_text(aes(label = scales::percent(Freq,accuracy=acc)), vjust = -0.5, size = 3) +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
      scale_y_continuous(labels = scales::percent) + labs(y = 'Percentage of observations', x = var) + 
      ggtitle(paste('Distribution of',var))    
  }
}


# Performs uni-variate analysis on continuous data
univ.cont <- function(ds,var,log.transform=F, acc=1){
  ds <- as.data.frame(ds[,var])
  names(ds) <- 'var'
  var.title <- var
  if(log.transform==T) {
    ds$var <- ifelse(ds$var ==0, 0, ifelse(ds$var<0, -log10(abs(ds$var)), log10(ds$var)))
    var.title <- paste('log10 of',var)
  }
  p1 <- ds %>% ggplot(aes(var))+geom_histogram(fill = "#FF7F24")+ ggtitle(paste('Histogram for', var.title))
  p2 <- ds %>% ggplot(aes(y=var))+geom_boxplot(fill = "#FF7F24")+ ggtitle(paste('Boxplot for', var.title))
  
  x <- seq(0.99,1,0.001)
  y <- quantile(ds$var,x)
  temp <- data.frame(x,y)
  p3 <- temp %>% ggplot(aes(x=x, y = y)) + geom_line() + geom_point() + 
    geom_text(label = paste0('(',x,',',round(y,acc),')'), hjust=1.5, size = 3)+
    ggtitle(paste('Values of higher percentiles for', var.title))
  
  x <- seq(0,0.01,0.001)
  y <- quantile(ds$var,x)
  temp <- data.frame(x,y)
  p4 <- temp %>% ggplot(aes(x=x, y = y)) + geom_point() + geom_line()+
    geom_text(label = paste0('(',x,',',round(y,acc),')'), hjust=-.5, size = 3)+
    ggtitle(paste('Values of lower percentiles for', var.title))
  
  grid.arrange(p1, p2, p3, p4, nrow = 2, ncol =2)
}

# Performs bi variate analysis of categorical independent variable with dependent variable
biv.categ <- function(ds,var, dv='y', response = 'yes', sort  = T, acc = 1){
  ds <- ds[,c(var,dv)]
  names(ds) <- c('var','dv')
  ds <- as.data.frame(prop.table(table(ds),1))
  ds <- ds %>% filter(dv == response)
  
  if(sort == T){
    ds %>% ggplot(aes(x=reorder(var,Freq), y = Freq)) + geom_col(fill = "#FF7F24") + 
      geom_text(aes(label = scales::percent(Freq,accuracy=acc)), vjust = -0.5, size = 3) +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
      scale_y_continuous(labels = scales::percent) + labs(y = 'Percentage of observations', x = var) + 
      ggtitle(paste('Distribution of',var))
  }else{
    ds %>% ggplot(aes(x=var, y = Freq)) + geom_col(fill = "#FF7F24") + 
      geom_text(aes(label = scales::percent(Freq,accuracy=acc)), vjust = -0.5, size = 3) +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
      scale_y_continuous(labels = scales::percent) + labs(y = 'Percentage of observations', x = var) + 
      ggtitle(paste('Distribution of',var))    
  }
}

###################################################################################
# PART 3 - Data Preparation for Model Building
###################################################################################
# data.split splits the dataset into training & validation basis inputs - percentage of observation in training & choice of outlier treatment
data.split <- function(dt, train.percentage = 0.9, outlier.treatment = T){
  # Checking for R Version and using set.seed function appropriately
  if(as.numeric(R.Version()$major)==3 & as.numeric(R.Version()$minor) >5) set.seed(1, sample.kind="Rounding") else set.seed(1)
  if(outlier.treatment == T) 
    dt <- dt %>% filter(is.outlier==F)
  train_index <- sample(1:nrow(dt),nrow(dt)*train.percentage)
  train_set <- dt[train_index,]
  train_set <- train_set %>% select(-is.outlier)
  test_set <- dt[-train_index,]
  test_set <- test_set %>% select(-is.outlier)
  return(list(train=train_set,validation=test_set))
}

plot.chaid <- function(dt,m.chaid,S = 1000, P = 11.7, D){
  dt$chaid.y <- predict(m.chaid, newdata = dt)
  dt$chaid.node <- predict(m.chaid, newdata = dt, type = 'node')
  temp <- as.data.frame(table(dt[,c('chaid.node', 'y')]))
  temp <- temp %>% spread(y, Freq)
  colnames(temp) <- c('node','y_0','y_1')
  temp <- temp %>% mutate(size = y_0+y_1, proportion = round(100*y_1/size,1), node.selection = ifelse(size>=S & proportion>P,'Selected','Not Selected'))
  temp2 <- temp %>% mutate(size = y_0+y_1, proportion = round(100*y_1/size,1)) %>% 
    filter(size>=S & proportion>P) %>% arrange(size)
  
  p <- temp %>% ggplot(aes(x=size, y = proportion, color = node.selection)) + geom_point() +
    geom_text(data = temp2, 
              aes(x=size, y = proportion, 
                  label = paste0('(Node =',node,', Size = ', size, ', % of Y as 1 = ',proportion,')')), hjust = -0.1) + 
    theme(legend.position = c(0.9, 0.85)) +ggtitle('Node selection for dummy variable creation basis CHAID', paste('Max depth of tree =', D)) +
    labs(x = 'Size of node', y = 'Proportion of 1 in y')
  
  overall.size <- sum(temp2$size)
  overall.y_1 <- sum(temp2$y_1)
  overall.prop <- round(100*overall.y_1/overall.size,1)
  overall.result <- paste('Overall size = ',overall.size,' , Overall count of Y as 1 = ',overall.y_1,' , Overall proportion of 1 in Y = ', overall.prop)
  print(temp2)
  print(overall.result)
  print(p)
}


###################################################################################
# PART 4 - Model Building
###################################################################################

add.model.result <- function(n, type, param, cm.t, cm.v, df){
  at <- round(cm.t[9,2],3)
  ft <- round(cm.t[10,2],3)
  av <- round(cm.v[9,2],3)
  fv <- round(cm.v[10,2],3)
  #df[nrow(df)+1,] <- c(n, type, param, at, av, ft, fv)
  df[nrow(df),] <- c(n, type, param, at, av, ft, fv)
  return(df)
}

view.model.results <- function(sort=T){
  mr <- model.results %>% arrange(desc(F1.validation))
  if(sort==T){
    pander(mr, style='simple', split.table = 400)
  }else{
    pander(model.results, split.cell = 80, split.table = Inf)
  } 
}


# Scoring on training dataset
get.predictions <- function(model, train, validation){
  score.train <- predict(model, train, type="response")
  rocCurve   <- roc(response = train$y, predictor = score.train, levels = c(0,1))
  # Best threshold
  threshold <- as.numeric(coords(rocCurve,"best")[1])
  score.validation <- predict(model, validation, type="response")
  predictions <-ifelse(score.validation > threshold, 1, 0)
  return(predictions)
}

confusion.matrix <- function(a,p){
  tn <- sum(a==0 & p==0)
  fp <- sum(a==0 & p==1)
  fn <- sum(a==1 & p==0)
  tp <- sum(a==1 & p==1)
  
  sn <- tp/(tp+fn)
  sp <- tn/(tn+fp)
  
  rc <- sn
  pc <- tp/(tp+fp)
  
  ac <- (tp+tn)/(tp+fp+tn+fn)
  f1 <- 2*rc*pc/(rc+pc)
  
  metric <- c('True Negative', 'False Positive', 'False Negative', 'True Positive', 
              'Sensitivity', 'Specificity', 'Recall','Precision','Accuracy','F1')
  value <- c(tn,fp,fn,tp,sn,sp,rc,pc,ac,f1)
  results <- data.frame(Metric = metric, Value = value)
  
  print(table(Actual = a, Predicted = p))
  return(results)
}


generate.lift.chart.table <- function(y,score){
  score.segment <- cut2(score,g=10)
  levels(score.segment) = c(1:10)
  lift.chart <- as.data.frame(table(score.segment, y))
  lift.chart <- lift.chart %>% spread(y,Freq)
  names(lift.chart) <- c('Segment','y_0', 'y_1')
  lift.chart <- lift.chart %>% mutate(Segment = 11-as.integer(Segment)) %>% arrange(Segment) %>%  
    mutate(PercentY_1 = 100*y_1/(sum(y)), Random = 10, CummPercentY_1 = cumsum(PercentY_1), 
           CummRandom = cumsum(Random), KS = CummPercentY_1 - CummRandom) %>% rbind(rep(0,8))%>% arrange(Segment) 
  return(lift.chart)
}


