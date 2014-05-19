CS909  Week 10: Text classification, clustering and topic models
Author: Elena Kochkina

This code performs task of text classification on Reuters-21578 dataset.
Dataset can be obtained
http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz.

0 Load dataset to R
follow instructions at
http://www2.warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/reuters

Run this code in R iteratively

Task is completed in 6 'easy' steps:

1 Split the data

2 Preprocessing
auxiliary functions:
removePunctuationCustom(x) 
posTagStem(x)
Input  - object of type VCorpus
Ouput   - object of type VCorpus
main function of preprocessing step:
preprocessCorpus(corpus)
Input  - object of type VCorpus
Ouput   - object of type VCorpus
Content of documents changed(removeNumbers, tolower,removeWords, removePunctuationCustom, stripWhitespace, posTagStem).
Meta data preserved.
3 Feature selection and classifier comparisson on training data
evalClassifier(train_corpus,train_class,fSelectFunc,nFolds,nMaxFeats)
Input  - train_corpus  - object of type VCorpus
         train_class - data frame with columns of binary classes  
         fSelectFunc - function from package fSelect for evaluation importance of features
         nFolds - integer -  for nFolds cross-validation
         nMaxFeats - integer - maximum number of features
Output - Accuracy per Fold,  Mean Accuracy,St Dev Accuracy,Precisions per Fold,
       Mean Precisions, St Dev Precisions, Micro Ave Precisions, Recalls per Fold,
       Mean Recalls, St Dev Recalls, Micro Ave Recalls,F1 Measure,Mean F1 Measure
       St Dev F1 Measure, Micro Ave F1 Measure for SVM and Naive Bayes classifier
Should be called for each binary class.
To choose number of features:

plot(1:50, chisq_10$"NAIVE BAYES"$"Mean F1 Measure",type="o",pch=4,col="blue",
     xlab="Number of Features", ylab="Mean F1 Score",xlim=c(1,50),ylim=c(0,0.9))
lines(1:50, ig_10$"NAIVE BAYES"$"Mean F1 Measure",type="o",pch=1,col="red")
legend("right",inset=.05,c("NB CS","NB IG"),
       col=c("blue","red"),pch=c(4,1), lty=1,
       title=expression(italic("corn")),bty="n")
       
To compare classifiers:   
  
t.test(chisq_1$"SVM"$"Accuracy"[,50],
       chisq_1$"NAIVE BAYES"$"Accuracy"[,49],
       alternative="greater",paired=TRUE,conf.level=0.95)

4 Topic models
evalClassifierTM (train_corpus,train_class,nFolds,nTopics)
Input  - train_corpus  - object of type VCorpus
         train_class - data frame with columns of binary classes  
         fSelectFunc - function from package fSelect for evaluation importance of features
         nFolds - integer -  for nFolds cross-validation
         nTopics - integer - number of topics
Output is the same as in  evalClassifier. Should be called for each binary class.       
         
       
5 Classification of testing data with best performing features and classifier
classifySVM (train_corpus,train_class,test_corpus,test_class,fSelectFunc,nFeats)

Input - train_corpus
        train_class
        test_corpus
        test_class
        fSelectFunc
        nFeats - number of features chosen as a result of tests on training data from previous step
As classifier SVM was chosen.

Output - Accuracy,Precision,Recall,F1 Measure,Confusion Matrix, Best performing features,
         SVM(trained classifier), Full feature list and ranking(for clustering).
Should be called for each binary class

6 Clustering
a) K-means
b) Hierarchical Agglomerative clustering
c) Gaussian finite mixture model fitted by EM algorithm 

To plot using Principal components:

plot(prcomp(y)$x, col=fit3$cl,pch=20, cex=0.5)

 
 clustCMf (groups,classes)
 Input - groups (Important! colnames(groups2)<-"groups")
         classes 
 Output - analog of contingency table
 Contains quantity of instances of each class in each cluster normalized by number of instances in cluster
 

