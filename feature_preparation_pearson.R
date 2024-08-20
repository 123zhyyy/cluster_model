library(caret)

data <- read.csv("mordred_var.csv",row.names=1)
X <- data[1:342,1:1123]
y <- data["Class"]
y <- unlist(y)
class(y)
str(X)#X
#head(X)
str(y)#y
#print(y)
#head(y)


correlations<-cor(X)
CalStrongCor<-function(x){
  cor_result<-as.data.frame(x)
  cor<-data.frame(col1=1,name1=2,col2=3,name2=4,cor=5)
  for(i in 1:(ncol(cor_result)-1)){
    for(j in (i+1):ncol(cor_result)){
      if(abs(cor_result[i,j]>0.9)){       
        ci<-c(i,names(cor_result)[i],j,names(cor_result)[j],cor_result[i,j]) ;
        cor<-rbind(cor,ci) ;
      }       
    }
  }
  return  (cor)   
}

cor<-CalStrongCor(correlations)
head(cor[-1,])
nrow(cor[-1,])


highCorr<-findCorrelation(correlations,cutoff=0.9)
highCorr
length(highCorr)
names(X[highCorr])

filtered_X<-X[,-highCorr]
write.csv(filtered_X, file = "mordred_pearson.csv", row.names = FALSE)