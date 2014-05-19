CS909  Week 10: Text classification, clustering and topic models
Elena Kochkina

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
