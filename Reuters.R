# Require necessary packages
require(tm)
require(tm.corpus.Reuters21578)
require(slam)
require(NLP)
require(openNLP)
require(openNLPdata)
require(rJava)
require(SnowballC)
require(modeltools)
require(topicmodels)
require(e1071)
require(caret)
require(FSelector)
require(mclust)
#################
# Split the data#
#################
data(Reuters21578)
rt <- Reuters21578

trainingIndicies=NULL
testingIndicies=NULL

#ModApte split + check content of "Topics"
for (i in 1:length(rt)){
  if(tm::meta(rt[[i]],tag="TOPICS")=="YES"){  
    if(length(tm::meta(rt[[i]],tag="Topics")) >0){    
      if(tm::meta(rt[[i]],tag="LEWISSPLIT")=="TRAIN"){    
        trainingIndicies=c(trainingIndicies,i)
      }
      if(tm::meta(rt[[i]],tag="LEWISSPLIT")=="TEST"){    
        testingIndicies=c(testingIndicies,i)
      }
    }
  }
}
#Getting rid of empty docs
dtm <- DocumentTermMatrix(rt)
dtmRowSums <- as.vector(rollup(dtm, 2, na.rm=TRUE, FUN = sum))
emptyDocIndicies <- which(dtmRowSums==0)

trainingIndicies=trainingIndicies[-sort(unique(match(emptyDocIndicies,trainingIndicies)))]
testingIndicies=testingIndicies[-sort(unique(match(emptyDocIndicies,testingIndicies)))]

trainC <- rt[trainingIndicies]
testC <- rt[testingIndicies]

# combinedCorpus for clustering
combinedIndices <- c(trainingIndicies,testingIndicies)
combinedCorpus <- rt[combinedIndices]
combinedCorpus<-preprocessCorpus(combinedCorpus)
#################
# Preprocessing #
#################
removePunctuationCustom <- function(x){    
  x <- gsub("[[:punct:]]+", " ", x)
}
posTagStem <- function(x, language = Language(x)) {
  mainFunc <- function(x) {
    z1 <- as.String(x)
    
    a2 <- annotate(x, list(Maxent_Sent_Token_Annotator(), Maxent_Word_Token_Annotator()))
    a3 <- annotate(x, Maxent_POS_Tag_Annotator(), a2) 
    gc()   #R garbage collection to fix Maxent_..._Annotator() memory problems
    
    a3w <- subset(a3,type=="word")
    tags <- sapply(a3w$features,`[[`,"POS")
    a3wstem <- stemDocument(z1[a3w])   #using SnowballC
    
    paste(sprintf("%s/%s",a3wstem,tags), collapse=" ")
  }
  
  s <- unlist(lapply(x, mainFunc))
  Content(x) <- if (is.character(s)) s else ""
  x
}
preprocessCorpus <- function(corpus) {
 
  corpus <- tm_map(corpus, removeNumbers)
  corpus <- tm_map(corpus, tolower)
  corpus <- tm_map(corpus, removeWords, stopwords("SMART"))
  corpus <- tm_map(corpus, removePunctuationCustom)
  corpus <- tm_map(corpus, stripWhitespace)
  corpus <- tm_map(corpus, posTagStem)
}

# Preprocess training, testing and combined corpus' 
trainC <- preprocessCorpus(trainC)
testC <- preprocessCorpus(testC)
combinedCorpus <- preprocessCorpus(combinedCorpus)
###############################################################################################################
# Feature selection and classifier 10 fold cross-validation, calculating performancwe metrics on training data#
###############################################################################################################
classNames=c("earn","acq","money-fx","grain","crude",
             "trade","interest","ship","wheat","corn")

TrainTopics=matrix(data=0,nrow=length(trainC),ncol=length(classNames),
                       dimnames=list(c(1:length(trainC)),classNames))
for (i in 1:length(trainC)){
  TrainTopics[i,match(tm::meta(trainC[[i]],tag="Topics"),classNames)]=1
}

testingClasses=matrix(data=0,nrow=length(testC),ncol=length(classNames),
                      dimnames=list(c(1:length(testC)),classNames))
for (i in 1:length(testC)){
  testingClasses[i,match(tm::meta(testC[[i]],tag="Topics"),classNames)]=1
}

# Function to perform x-val for given classifier and given feature selection method and class
evalClassifier <- function(train_corpus,train_class,fSelectFunc,nFolds,nMaxFeats){
  # Function to compute matrix column standard deviations
  colSd <- function(x, na.rm=TRUE) {
    if (na.rm) {
      n <- colSums(!is.na(x))
    } else {
      n <- nrow(x)
    }
    colVar <- colMeans(x*x, na.rm=na.rm) - (colMeans(x, na.rm=na.rm))^2
    return(sqrt(colVar * n/(n-1)))
  }
  
  nb_accuracy <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  nb_precision <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  nb_recall <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  nb_f1 <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  
  svm_accuracy <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  svm_precision <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  svm_recall <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  svm_f1 <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  
  nb_TP <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  nb_TPplusFP <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  nb_TPplusFN <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  
  svm_TP <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  svm_TPplusFP <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  svm_TPplusFN <- matrix(data=0,nrow=nFolds,ncol=nMaxFeats,dimnames=list(1:nFolds,1:nMaxFeats))
  
  numDocs <- length(train_corpus)
  reordered <- sample(numDocs)
  folds <- cut(1:numDocs,breaks=nFolds,labels=F)
  
  # Looping over number of folds
  for (i in 1:nFolds){
       
    # Create Document-Term Matrix for documents not in this fold - training set.
    train_dtm <- DocumentTermMatrix(train_corpus[reordered[folds!=i]]
                                    ,control=list(weighting=weightTfIdf))
    
    # Create Document-Term Matrix for documents in this fold only - testing set.
    test_dtm <- DocumentTermMatrix(train_corpus[reordered[folds==i]]
                                   ,control=list(weighting=weightTfIdf))
    
                                             ceiling(0.01*dim(train_dtm)[1]))]
    
    
    dtm_col_sums <- as.vector(rollup(train_dtm, 1, na.rm=TRUE, FUN = sum))
    reduced_train_dtm <- train_dtm[,sort(match(sort(dtm_col_sums,decreasing=TRUE)
                                               [1:floor(0.05*length(train_dtm$dimnames$Terms))],dtm_col_sums))]
    
    # Turn into form that FSelector package works well with
    d1 <- as.data.frame(as.matrix(reduced_train_dtm))
    d1$class <- train_class[reordered[folds!=i]]
    
    # Compute feature relevance measure for each term
    spisok <- fSelectFunc(class~.,d1)
    
    for (j in 2:nMaxFeats){
      # (Sort? and) Select the most highly relevant features (in some defined way...)
      #sorted <- spisok[order(spisok[,"attr_importance"]), , drop=FALSE]
      subset <- cutoff.k(spisok,j)
      
      # Train model for class j using train_dtm and jth column of train_class
      nb_model <- naiveBayes(d1[,subset],as.factor(d1[,"class"]))
      
      # Find features not in testing dtm - v. important to do first for indexing out others!
      missingFeatureIndicies <- which(is.na(match(subset,Terms(test_dtm)))==1)
      if(length(missingFeatureIndicies) >0){
        subset <- subset[-missingFeatureIndicies]
      }
      
      # SVM must train on (potentially) updated subset, so the space dimensions are consistent
      # with the testing set, and the prediction function can work - no such problem for NB.
      svm_model <- svm(d1[,subset],as.factor(d1[,"class"]))
      
      # Subset the terms to look for in the testing set
      featureSelectedTestDtm <- as.data.frame(as.matrix(test_dtm[,subset]))
      
      # Predict if the documents in the test set contain the ith topic or not
      nb_results <- predict(nb_model,featureSelectedTestDtm)
      svm_results <- predict(svm_model,featureSelectedTestDtm)
      
      # Compute classification performance statistics
      nb_CM <- confusionMatrix(nb_results,train_class[reordered[folds==i]],positive="1")
      nb_accuracy[i,j] <- as.numeric(nb_CM$overall["Accuracy"])
      nb_precision[i,j] <- as.numeric(nb_CM$byClass["Pos Pred Value"])
      nb_recall[i,j] <- as.numeric(nb_CM$byClass["Sensitivity"])
      nb_f1[i,j] <- as.numeric((2*nb_recall[i,j]*nb_precision[i,j])/
                                 (nb_recall[i,j]+nb_precision[i,j]))
      nb_TP[i,j] <- nb_CM$table["1","1"]
      nb_TPplusFP[i,j] <- nb_TP[i,j] + nb_CM$table["1","0"]
      nb_TPplusFN[i,j] <- nb_TP[i,j] + nb_CM$table["0","1"]
      
      svm_CM <- confusionMatrix(svm_results,train_class[reordered[folds==i]],positive="1")
      svm_accuracy[i,j] <- as.numeric(svm_CM$overall["Accuracy"])
      svm_precision[i,j] <- as.numeric(svm_CM$byClass["Pos Pred Value"])
      svm_recall[i,j] <- as.numeric(svm_CM$byClass["Sensitivity"])
      svm_f1[i,j] <- as.numeric((2*svm_recall[i,j]*svm_precision[i,j])/
                                  (svm_recall[i,j]+svm_precision[i,j]))
      svm_TP[i,j] <- svm_CM$table["1","1"]
      svm_TPplusFP[i,j] <- svm_TP[i,j] + svm_CM$table["1","0"]
      svm_TPplusFN[i,j] <- svm_TP[i,j] + svm_CM$table["0","1"]
    }
  }
  
  Out <- list()
  N.B <- list()
  S.V.M <- list()
  
  N.B[[paste("Accuracy per Fold")]] <- nb_accuracy
  N.B[[paste("Mean Accuracy")]] <- colMeans(nb_accuracy)
  N.B[[paste("St Dev Accuracy")]] <- colSd(nb_accuracy)
  N.B[[paste("Precisions per Fold")]] <- nb_precision
  N.B[[paste("Mean Precisions")]] <- colMeans(nb_precision)
  N.B[[paste("St Dev Precisions")]] <- colSd(nb_precision)
  nb_micro_precision <- colSums(nb_TP)/colSums(nb_TPplusFP)
  N.B[[paste("Micro Ave Precisions")]] <- nb_micro_precision
  N.B[[paste("Recalls per Fold")]] <- nb_recall
  N.B[[paste("Mean Recalls")]] <- colMeans(nb_recall)
  N.B[[paste("St Dev Recalls")]] <- colSd(nb_recall)
  nb_micro_recall <- colSums(nb_TP)/colSums(nb_TPplusFN)
  N.B[[paste("Micro Ave Recalls")]] <- nb_micro_recall
  N.B[[paste("F1 Measure")]] <- nb_f1
  N.B[[paste("Mean F1 Measure")]] <- colMeans(nb_f1)
  N.B[[paste("St Dev F1 Measure")]] <- colSd(nb_f1)
  N.B[[paste("Micro Ave F1 Measure")]] <- (2*nb_micro_precision*nb_micro_recall)/
    (nb_micro_precision+nb_micro_recall)
  
  S.V.M[[paste("Accuracy per Fold")]] <- svm_accuracy
  S.V.M[[paste("Mean Accuracy")]] <- colMeans(svm_accuracy)
  S.V.M[[paste("St Dev Accuracy")]] <- colSd(svm_accuracy)
  S.V.M[[paste("Precisions per Fold")]] <- svm_precision
  S.V.M[[paste("Mean Precisions")]] <- colMeans(svm_precision)
  S.V.M[[paste("St Dev Precisions")]] <- colSd(svm_precision)
  svm_micro_precision <- colSums(svm_TP)/colSums(svm_TPplusFP)
  S.V.M[[paste("Micro Ave Precisions")]] <- svm_micro_precision
  S.V.M[[paste("Recalls per Fold")]] <- svm_recall
  S.V.M[[paste("Mean Recalls")]] <- colMeans(svm_recall)
  S.V.M[[paste("St Dev Recalls")]] <- colSd(svm_recall)
  svm_micro_recall <- colSums(svm_TP)/colSums(svm_TPplusFN)
  S.V.M[[paste("Micro Ave Recalls")]] <- svm_micro_recall
  S.V.M[[paste("F1 Measure")]] <- svm_f1
  S.V.M[[paste("Mean F1 Measure")]] <- colMeans(svm_f1)
  S.V.M[[paste("St Dev F1 Measure")]] <- colSd(svm_f1)
  S.V.M[[paste("Micro Ave F1 Measure")]] <- (2*svm_micro_precision*svm_micro_recall)/
    (svm_micro_precision+svm_micro_recall)
  
  Out[[paste("NAIVE BAYES")]] <- N.B
  Out[[paste("SVM")]] <- S.V.M
  
  return(Out)
}

t.test(chisq_1$"SVM"$"Accuracy"[,50],
       chisq_1$"NAIVE BAYES"$"Accuracy"[,49],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(ig_2$"SVM"$"Accuracy"[,49],
       chisq_2$"NAIVE BAYES"$"Accuracy"[,49],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(ig_3$"SVM"$"Accuracy"[,34],
       ig_3$"NAIVE BAYES"$"Accuracy"[,9],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(ig_4$"SVM"$"Accuracy"[,11],
       chisq_4$"NAIVE BAYES"$"Accuracy"[,2],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(chisq_5$"SVM"$"Accuracy"[,18],
       chisq_5$"NAIVE BAYES"$"Accuracy"[,4],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(ig_6$"SVM"$"Accuracy"[,23],
       ig_6$"NAIVE BAYES"$"Accuracy"[,4],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(ig_7$"SVM"$"Accuracy"[,28],
       ig_7$"NAIVE BAYES"$"Accuracy"[,12],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(ig_8$"SVM"$"Accuracy"[,10],
       chisq_8$"NAIVE BAYES"$"Accuracy"[,5],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(chisq_9$"SVM"$"Accuracy"[,2],
       ig_9$"NAIVE BAYES"$"Accuracy"[,2],
       alternative="greater",paired=TRUE,conf.level=0.95)
t.test(chisq_10$"SVM"$"Accuracy"[,5],
       ig_10$"NAIVE BAYES"$"Accuracy"[,2],
       alternative="greater",paired=TRUE,conf.level=0.95)

# It's time for Topic models!
evalClassifier <- function(train_corpus,train_class,nFolds,nTopics){
  # Function to compute matrix column standard deviations
  colSd <- function(x, na.rm=TRUE) {
    if (na.rm) {
      n <- colSums(!is.na(x))
    } else {
      n <- nrow(x)
    }
    colVar <- colMeans(x*x, na.rm=na.rm) - (colMeans(x, na.rm=na.rm))^2
    return(sqrt(colVar * n/(n-1)))
  }
  
  nb_accuracy <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  nb_precision <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  nb_recall <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  nb_f1 <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  
  svm_accuracy <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  svm_precision <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  svm_recall <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  svm_f1 <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  
  nb_TP <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  nb_TPplusFP <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  nb_TPplusFN <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  
  svm_TP <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  svm_TPplusFP <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  svm_TPplusFN <- matrix(data=0,nrow=nFolds,ncol=1,dimnames=list(1:nFolds,1:1))
  
  numDocs <- length(train_corpus)
  reordered <- sample(numDocs)
  folds <- cut(1:numDocs,breaks=nFolds,labels=F)
  
  # Looping over number of folds
  for (i in 1:nFolds){
    # Output current progress
    print(i)
    flush.console()
    
    # Create Document-Term Matrix for documents not in this fold - training set.
    train_dtm <- DocumentTermMatrix(train_corpus[reordered[folds!=i]]
                                    ,control=list(weighting=weightTf))
    
    # Create Document-Term Matrix for documents in this fold only - testing set.
    test_dtm <- DocumentTermMatrix(train_corpus[reordered[folds==i]]
                                   ,control=list(weighting=weightTf))
    
    
    reduced_train_dtm <- train_dtm[,findFreqTerms(train_dtm,
                                                  ceiling(0.01*dim(train_dtm)[1]))]
            
    # Time for LDA
    LDA_train<-LDA(reduced_train_dtm,nTopics,method="Gibbs")
    posterior(LDA_train,reduced_train_dtm)
    
    posterior_train<-posterior(LDA_train,reduced_train_dtm)
    posterior_test<-posterior(LDA_train,test_dtm)
    
    
    d1<-as.data.frame(as.matrix(posterior_train$topics))
    d2<-as.data.frame(as.matrix(posterior_test$topics))
    
    d1$class <- train_class[reordered[folds!=i]]
    
    j=1
        
    # Train model for class j using train_dtm and jth column of train_class
    nb_model <- naiveBayes(d1[,-(nTopics+1)],as.factor(d1[,"class"]))
    svm_model <- svm(d1[,-(nTopics+1)],as.factor(d1[,"class"]))
    
    # Predict if the documents in the test set contain the ith topic or not
    nb_results <- predict(nb_model,d2)
    svm_results <- predict(svm_model,d2)
    
    # Compute classification performance statistics
    nb_CM <- confusionMatrix(nb_results,train_class[reordered[folds==i]],positive="1")
    nb_accuracy[i,j] <- as.numeric(nb_CM$overall["Accuracy"])
    nb_precision[i,j] <- as.numeric(nb_CM$byClass["Pos Pred Value"])
    nb_recall[i,j] <- as.numeric(nb_CM$byClass["Sensitivity"])
    nb_f1[i,j] <- as.numeric((2*nb_recall[i,j]*nb_precision[i,j])/
                               (nb_recall[i,j]+nb_precision[i,j]))
    nb_TP[i,j] <- nb_CM$table["1","1"]
    nb_TPplusFP[i,j] <- nb_TP[i,j] + nb_CM$table["1","0"]
    nb_TPplusFN[i,j] <- nb_TP[i,j] + nb_CM$table["0","1"]
    
    svm_CM <- confusionMatrix(svm_results,train_class[reordered[folds==i]],positive="1")
    svm_accuracy[i,j] <- as.numeric(svm_CM$overall["Accuracy"])
    svm_precision[i,j] <- as.numeric(svm_CM$byClass["Pos Pred Value"])
    svm_recall[i,j] <- as.numeric(svm_CM$byClass["Sensitivity"])
    svm_f1[i,j] <- as.numeric((2*svm_recall[i,j]*svm_precision[i,j])/
                                (svm_recall[i,j]+svm_precision[i,j]))
    svm_TP[i,j] <- svm_CM$table["1","1"]
    svm_TPplusFP[i,j] <- svm_TP[i,j] + svm_CM$table["1","0"]
    svm_TPplusFN[i,j] <- svm_TP[i,j] + svm_CM$table["0","1"]
  }
  
  
  Out <- list()
  N.B <- list()
  S.V.M <- list()
  
  N.B[[paste("Accuracy per Fold")]] <- nb_accuracy
  N.B[[paste("Mean Accuracy")]] <- colMeans(nb_accuracy)
  N.B[[paste("St Dev Accuracy")]] <- colSd(nb_accuracy)
  N.B[[paste("Precisions per Fold")]] <- nb_precision
  N.B[[paste("Mean Precisions")]] <- colMeans(nb_precision)
  N.B[[paste("St Dev Precisions")]] <- colSd(nb_precision)
  nb_micro_precision <- colSums(nb_TP)/colSums(nb_TPplusFP)
  N.B[[paste("Micro Ave Precisions")]] <- nb_micro_precision
  N.B[[paste("Recalls per Fold")]] <- nb_recall
  N.B[[paste("Mean Recalls")]] <- colMeans(nb_recall)
  N.B[[paste("St Dev Recalls")]] <- colSd(nb_recall)
  nb_micro_recall <- colSums(nb_TP)/colSums(nb_TPplusFN)
  N.B[[paste("Micro Ave Recalls")]] <- nb_micro_recall
  N.B[[paste("F1 Measure")]] <- nb_f1
  N.B[[paste("Mean F1 Measure")]] <- colMeans(nb_f1)
  N.B[[paste("St Dev F1 Measure")]] <- colSd(nb_f1)
  N.B[[paste("Micro Ave F1 Measure")]] <- (2*nb_micro_precision*nb_micro_recall)/
    (nb_micro_precision+nb_micro_recall)
  
  S.V.M[[paste("Accuracy per Fold")]] <- svm_accuracy
  S.V.M[[paste("Mean Accuracy")]] <- colMeans(svm_accuracy)
  S.V.M[[paste("St Dev Accuracy")]] <- colSd(svm_accuracy)
  S.V.M[[paste("Precisions per Fold")]] <- svm_precision
  S.V.M[[paste("Mean Precisions")]] <- colMeans(svm_precision)
  S.V.M[[paste("St Dev Precisions")]] <- colSd(svm_precision)
  svm_micro_precision <- colSums(svm_TP)/colSums(svm_TPplusFP)
  S.V.M[[paste("Micro Ave Precisions")]] <- svm_micro_precision
  S.V.M[[paste("Recalls per Fold")]] <- svm_recall
  S.V.M[[paste("Mean Recalls")]] <- colMeans(svm_recall)
  S.V.M[[paste("St Dev Recalls")]] <- colSd(svm_recall)
  svm_micro_recall <- colSums(svm_TP)/colSums(svm_TPplusFN)
  S.V.M[[paste("Micro Ave Recalls")]] <- svm_micro_recall
  S.V.M[[paste("F1 Measure")]] <- svm_f1
  S.V.M[[paste("Mean F1 Measure")]] <- colMeans(svm_f1)
  S.V.M[[paste("St Dev F1 Measure")]] <- colSd(svm_f1)
  S.V.M[[paste("Micro Ave F1 Measure")]] <- (2*svm_micro_precision*svm_micro_recall)/
    (svm_micro_precision+svm_micro_recall)
  
  Out[[paste("NAIVE BAYES")]] <- N.B
  Out[[paste("SVM")]] <- S.V.M
  
  return(Out)
}
ptm <- proc.time()

TopicModelResult=evalClassifier(trainC,TrainTopics[,classNames[1]],10,25)

proc.time() - ptm
###############################################################################
# Classification of testing data with best performing features and classifier#
##############################################################################
classifySVM <- function(train_corpus,train_class,test_corpus,test_class,fSelectFunc,nFeats){
  # Create Document-Term Matrix for documents in training set.
  train_dtm <- DocumentTermMatrix(train_corpus,control=list(weighting=weightTfIdf))
  
  # Create Document-Term Matrix for documents in testing set.
  test_dtm <- DocumentTermMatrix(test_corpus,control=list(weighting=weightTfIdf))
  
  # First dimensionality reduction based on ranking via TfIdf - takes 'top' 5% of term-summed TdIdf
  # some assumptions naturally but a good start, and neccessary for FSelector to handle in RAM! 
  dtm_col_sums <- as.vector(rollup(train_dtm, 1, na.rm=TRUE, FUN = sum))
  reduced_train_dtm <- train_dtm[,sort(match(sort(dtm_col_sums,decreasing=TRUE)
                                             [1:floor(0.05*length(train_dtm$dimnames$Terms))],dtm_col_sums))]
  
  # Turn into form that FSelector package works well with
  d1 <- as.data.frame(as.matrix(reduced_train_dtm))
  d1$class <- train_class
  
  # Compute feature relevance measure for each term
  spisok <- fSelectFunc(class~.,d1)
  
  # (Sort? and) Select the most highly relevant features (in some defined way...)
  #sorted <- spisok[order(spisok[,"attr_importance"]), , drop=FALSE]
  subset <- cutoff.k(spisok,nFeats)
  
  # Find features not in testing dtm - v. important to do first for indexing out others!
  missingFeatureIndicies <- which(is.na(match(subset,Terms(test_dtm)))==1)
  if(length(missingFeatureIndicies) >0){
    subset <- subset[-missingFeatureIndicies]
  }
  
  # SVM must train on (potentially) updated subset, so the space dimensions are consistent
  # with the testing set, and the prediction function can work.
  svm_model <- svm(d1[,subset],as.factor(d1[,"class"]))
  
  # Subset the terms to look for in the testing set
  featureSelectedTestDtm <- as.data.frame(as.matrix(test_dtm[,subset]))
  
  # Predict if the documents in the test set contain the topic or not
  svm_results <- predict(svm_model,featureSelectedTestDtm)
  
  # Compute classification performance statistics      
  svm_CM <- confusionMatrix(svm_results,test_class,positive="1")
  svm_accuracy <- as.numeric(svm_CM$overall["Accuracy"])
  svm_precision <- as.numeric(svm_CM$byClass["Pos Pred Value"])
  svm_recall <- as.numeric(svm_CM$byClass["Sensitivity"])
  svm_f1 <- as.numeric((2*svm_recall*svm_precision)/(svm_recall+svm_precision))
  
  Out <- list()
  S.V.M <- list()
  
  S.V.M[[paste("Accuracy")]] <- svm_accuracy
  S.V.M[[paste("Precision")]] <- svm_precision
  S.V.M[[paste("Recall")]] <- svm_recall
  S.V.M[[paste("F1 Measure")]] <- svm_f1
  S.V.M[[paste("Confusion Matrix")]] <- svm_CM
  
  Out[[paste("Best performing features")]] <- subset
  Out[[paste("SVM")]] <- S.V.M
  Out[[paste("Full feature list and ranking")]] <- spisok[order(spisok[,"attr_importance"])
                                                                 , , drop=FALSE]
  
  return(Out)
}

ptm <- proc.time()
testing1=classifySVM(trainC,TrainTopics[,classNames[1]]
                     ,testC,testingClasses[,classNames[1]],chi.squared,50)
testing2=classifySVM(trainC,TrainTopics[,classNames[2]]
                     ,testC,testingClasses[,classNames[2]],information.gain,49)
testing3=classifySVM(trainC,TrainTopics[,classNames[3]]
                     ,testC,testingClasses[,classNames[3]],information.gain,34)
testing4=classifySVM(trainC,TrainTopics[,classNames[4]]
                     ,testC,testingClasses[,classNames[4]],information.gain,11)
testing5=classifySVM(trainC,TrainTopics[,classNames[5]]
                     ,testC,testingClasses[,classNames[5]],chi.squared,18)
ptm <- proc.time()
testing6=classifySVM(trainC,TrainTopics[,classNames[6]]
                     ,testC,testingClasses[,classNames[6]],information.gain,23)
testing7=classifySVM(trainC,TrainTopics[,classNames[7]]
                     ,testC,testingClasses[,classNames[7]],information.gain,28)
testing8=classifySVM(trainC,TrainTopics[,classNames[8]]
                     ,testC,testingClasses[,classNames[8]],information.gain,10)
testing9=classifySVM(trainC,TrainTopics[,classNames[9]]
                     ,testC,testingClasses[,classNames[9]],chi.squared,2)
testing10=classifySVM(trainC,TrainTopics[,classNames[10]]
                      ,testC,testingClasses[,classNames[10]],chi.squared,5)
proc.time() - ptm

#############
# Clustering#
#############
classNames=c("earn","acq","money-fx","grain","crude",
             "trade","interest","ship","wheat","corn")
allClasses=matrix(data=0,nrow=length(combinedCorpus),ncol=length(classNames),
                  dimnames=list(c(1:length(combinedCorpus)),classNames))
for (i in 1:length(combinedCorpus)){
  allClasses[i,match(tm::meta(combinedCorpus[[i]],tag="Topics"),classNames)]=1
}

DTM_combC<-DocumentTermMatrix(combinedCorpus)
y<-DTM_combC[,bestPerfFeats]
y<-as.data.frame(as.matrix(y))

# Hierarchical Agglomerative 
distance<- dist(y, method = "euclidean") # or binary,canberra, maximum, manhattan
fit1 <- hclust(distance, method="ward")
groups <- cutree(fit1, k=10) # cut tree into 5 clusters
groups1 <- as.data.frame(groups)
plot(fit1, labels = NULL, hang = 0.1,
     axes = TRUE, frame.plot = FALSE, ann = TRUE,
     main = "Cluster Dendrogram",
     sub = NULL, xlab = NULL, ylab = "Height") # display dendogram
rect.hclust(fit1, k=10, border="red")

plot(prcomp(y)$x, col=groups, pch=20, cex=0.5,xlim =c(-3.5,10.5),ylim=c(-10,5))


# K-means
wssplot <- function(data, nc=15, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}
#df <- scale(y)
wssplot(y) 

fit2 <- kmeans(y, 10)
plot(prcomp(y)$x, col=fit2$cl,pch=20, cex=0.5,xlim =c(-3.5,10.5),ylim=c(-10,5))


# EM clustering

library(mclust)

ptm <- proc.time()
fit3 <- Mclust(y,G=10)
proc.time() - ptm

plot(prcomp(y)$x, col=fit3$cl,pch=20, cex=0.5,xlim =c(-3.5,10.5),ylim=c(-10,5))
#summary(fit3) # display the best model

clustCMf <- function(groups,classes){
  clustCM<-matrix(0,10,12)
  colnames(clustCM)=c("earn","acq","money-fx","grain","crude","trade","interest","ship","wheat","corn","others","total")
  
  for(i in 1:dim(classes)[1])
  {
    if (sum(classes[i,])==0){clustCM[groups$groups[i],11]=clustCM[groups$groups[i],11]+1; } 
    else
    {clustCM[groups$groups[i],1:10]= clustCM[groups$groups[i],1:10]+classes[i,1:10]}
    
  }
  
  for(i in 1:10)
  {
    clustCM[i,1:11]=clustCM[i,1:11]/length(which(groups$groups==i))
    clustCM[i,12]=length(which(groups$groups==i))
  }
  
  return (clustCM)
}
#cbind(TrainTopics,rep(0, length(TrainTopics) ))
table1<- clustCMf(groups1,allClasses)
groups2<-as.data.frame(fit2$cl)
colnames(groups2)<-"groups"
table2<- clustCMf(groups2,allClasses)
groups3<-as.data.frame(fit3$cl)
colnames(groups3)<-"groups"
table3<- clustCMf(groups3,allClasses)
