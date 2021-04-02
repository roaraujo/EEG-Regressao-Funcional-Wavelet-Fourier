require(fda.usc)
library(dplyr)

setwd("C:\\Users\\Rodrigo Araujo\\Documents\\IME-USP\\IC - EEG\\Alcoholism\\Data_clear\\")

alcoholism_train = read.csv("Alcoholism_train.csv", sep = ',')
alcoholism_test = read.csv("Alcoholism_test.csv", sep = ',')


s2_match_train <- alcoholism_train %>% filter(matching.condition=="S2 match") %>% dplyr::select(subject.identifier, time, sensor.value, channel)
s2_match_test <- alcoholism_test %>% filter(matching.condition=="S2 match") %>% dplyr::select(subject.identifier, time, sensor.value, channel)


s2_match_train$sensor.value <- as.numeric(s2_match_train$sensor.value)
s2_match_train$channel <- as.numeric(s2_match_train$time)

s2_match_test$sensor.value <- as.numeric(s2_match_test$sensor.value)
s2_match_test$channel <- as.numeric(s2_match_test$time)



#Transforma duas vari?veis em Functional Data
x.v2.fdata <- fdata(s2_match_train$sensor.value)
x.v3.fdata <- fdata(s2_match_train$channel)


s2_match_train$subject.identifier <-ifelse(s2_match_train$subject.identifier=="a",1,0)
s2_match_test$subject.identifier <-ifelse(s2_match_test$subject.identifier=="a",1,0)

y_train <- rep(ifelse(s2_match_train$subject.identifier == 'a', 1, 0))
y_test <- rep(ifelse(s2_match_test$subject.identifier == 'a', 1, 0))



#Cria as bases Spline/Fourier
ldata.train=list("df"=as.data.frame(y_train),"x1"=x.v2.fdata,"x2" =x.v3.fdata)  
basis.x1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:4); basis.x2=create.fdata.basis(x.v3.fdata,type.basis="bspline", l=1:4) 
basis.b1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:4); basis.b2=create.fdata.basis(x.v3.fdata,type.basis="bspline", l=1:4)   
basis.x=list("x1"=basis.x1, "x2"=basis.x2)                                                   
basis.b=list("x1"=basis.b1, "x2"=basis.b2)      

#Roda o GLM com a bas escolhida
res.basis=fregre.glm(y_train~x1+x2,ldata.train,family=binomial,basis.x=basis.x,basis.b=basis.b)    
summary(res.basis)

#Avalia performance (acur?cia) nas bases de treino e teste
ldata.test=list("df"=as.data.frame(y_test),"x1"=fdata(s2_match_test$sensor.value),"x2" =fdata(s2_match_test$time))  


predictions <- predict(res.basis, type = "response")
round_preds <- ifelse(predictions < 0.55, 0, 1)
table(round_preds, s2_match_train$subject.identifier)


table(y_train, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1)) #equivalente: table(y_train, ifelse(fitted(res.basis) < 0.5, 0, 1))
table(y_test, ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1))

#Teste fPCA, passos an?logos aos acima para Spline/Fourier
basis.pc1=create.pc.basis(x.v2.fdata, l = 1:5)
basis.x.pc=list("x1"=basis.pc1)            
res.pc=fregre.glm(y_train~x1,ldata.train,family=binomial,basis.x=basis.x.pc)     
summary(res.pc)
table(y_train, ifelse(predict(res.pc, newx = ldata.train) < 0.5, 0, 1)) #equivalente: table(y_train, ifelse(fitted(res.basis) < 0.5, 0, 1))
table(y_test, ifelse(predict(res.pc, newx = ldata.test) < 0.5, 0, 1))


### Implementa um cross-validation com K folds
k = 3
folds <- rep(1:k, ceiling(length(y_train)/k))[1:length(y_train)]#cut(seq(1,nrow(ldata.train$df)),breaks=3,labels=FALSE)

#definir quantos valores diferentes de qtd.bases ser?o testadas dentro do loop
for(number.basis in 2*(3:7)){
  acc = NULL
  for(i in 1:k){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    
    basis.x1=create.fdata.basis(fdata.deriv(x_train[-testIndexes,]$V2, nderiv=2),type.basis="bspline", l=1:number.basis)
    basis.x2=create.fdata.basis(fdata.deriv(x_train[-testIndexes,]$V3, nderiv=2),type.basis="bspline", l=1:number.basis) 
    basis.b1=create.fdata.basis(fdata.deriv(x_train[-testIndexes,]$V2, nderiv=2),type.basis="bspline", l=1:number.basis)
    basis.b2=create.fdata.basis(fdata.deriv(x_train[-testIndexes,]$V3, nderiv=2),type.basis="bspline", l=1:number.basis)   
    basis.x=list("x1"=basis.x1, "x2"=basis.x2); basis.b=list("x1"=basis.b1, "x2"=basis.b2) 
    
    ldata.train.cv=list("df"=as.data.frame(y_train[-testIndexes]),"x1"=fdata.deriv(x_train[-testIndexes,]$V2, nderiv=2),"x2" =fdata.deriv(x_train[-testIndexes,]$V3, nderiv=2))  
    ldata.valid.cv=list("df"=as.data.frame(y_train[testIndexes]),"x1"=fdata.deriv(x_train[testIndexes,]$V2, nderiv=2),"x2" =fdata.deriv(x_train[testIndexes,]$V3, nderiv=2))  
    res.basis.cv=fregre.glm(`y_train[-testIndexes]`~x1+x2,ldata.train.cv,family=binomial,basis.x=basis.x,basis.b=basis.b)    
    table.cv = table(y_train[testIndexes], ifelse(predict(res.basis.cv, newx = ldata.valid.cv) < 0.5, 0, 1))
    #print(summary(res.basis))
    #print(table.cv)
    acc = append(acc, table.cv %>% diag() %>% sum () / table.cv %>% sum())
  } 
  print(number.basis); print(mean(acc))
}
