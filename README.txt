CS909  Week 10: Text classification, clustering and topic models
Author: Elena Kochkina

This code performs task of text classification on Reuters-21578 dataset.
Dataset can be obtained
http://www.daviddlewis.com/resources/testcollections/reuters21578/reuters21578.tar.gz.

0 Load dataset to R
follow instructions at
http://www2.warwick.ac.uk/fac/sci/dcs/teaching/material/cs909/reuters

Run this code in R iteratively

1 Split the data

2 Preprocessing
removePunctuationCustom(x)
posTagStem(x)
preprocessCorpus(corpus)
3 Feature selection and classifier comparisson on training data
4 It's time for Topic models

how to plot

plot(1:50, chisq_10$"NAIVE BAYES"$"Mean F1 Measure",type="o",pch=4,col="blue",
     xlab="Number of Features", ylab="Mean F1 Score",xlim=c(1,50),ylim=c(0,0.9))
lines(1:50, ig_10$"NAIVE BAYES"$"Mean F1 Measure",type="o",pch=1,col="red")
legend("right",inset=.05,c("NB CS","NB IG"),
       col=c("blue","red"),pch=c(4,1), lty=1,
       title=expression(italic("corn")),bty="n")
       
5 Classification of testing data with best performing features and classifier
6 Clustering
