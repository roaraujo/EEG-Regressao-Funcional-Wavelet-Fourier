require(reticulate)
require(dplyr)
library(refund)
library(mgcv)
library(pROC)
library(caret)
require(fda.usc)
np <- import('numpy')


library(wavelets)
library(wavethresh)

library(stats)

plotaroc <- function(rocobj, titulo = "Curva ROC"){
  # Função que plota as curvas roc para os modelos ajustados 
  b <- which.max(rocobj$sensitivities + rocobj$specificities)
  best <- round(c(rocobj$thresholds[b],rocobj$specificities[b],rocobj$sensitivities[b]), 3)
  
  pROC::ggroc(rocobj, col = "red", alpha = 0.5, size = 0.5) + 
    theme_gray() + 
    ggtitle(titulo) + 
    geom_abline(intercept = 1, slope=1, linetype = "dashed") +
    labs(x="Especificidade", y = "Sensibilidade")  +
    geom_point(data = tibble(Sensibilidade = best[2],
                             Especificidade = best[3]),
               mapping = aes(x=Sensibilidade, y=Especificidade),
               col = "black") +
    geom_text(mapping =  aes(x = best[2] - 0.15,
                             y = best[3] - 0.05),
              label = paste( best[1], "(", best[2], ",", best[3], ")")) +
    geom_text(mapping = aes(x = 0.5,
                            y = 0.01),
              label = paste("AUC: ", round(rocobj$auc,3)))
}



x_train_py <- np$load('C:\\Users\\Rodrigo Araujo\\Documents\\IME-USP\\IC - EEG\\eeg_survey\\Datasets_clean\\SelfRegulationSCP1\\X_train.npy')
x_test_py <- np$load('C:\\Users\\Rodrigo Araujo\\Documents\\IME-USP\\IC - EEG\\eeg_survey\\Datasets_clean\\SelfRegulationSCP1\\X_test.npy')

y_train_py <- np$load('C:\\Users\\Rodrigo Araujo\\Documents\\IME-USP\\IC - EEG\\eeg_survey\\Datasets_clean\\SelfRegulationSCP1\\y_train.npy')
y_test_py <- np$load('C:\\Users\\Rodrigo Araujo\\Documents\\IME-USP\\IC - EEG\\eeg_survey\\Datasets_clean\\SelfRegulationSCP1\\y_test.npy')

x_train_melted <- as.array(x_train_py)
x_test_melted <- as.array(x_test_py)

dim(x_train_melted) <- c(268*6, 896)
dim(x_test_melted) <- c(293*6, 896)

x_train <- data.frame(id = 1:268)
for(i in 1:6){
  x_train[,(1+i)] <- x_train_melted[(1+(i-1)*268):(268+(i-1)*268),]
}

x_test <- data.frame(id = 1:293)
for(i in 1:6){
  x_test[,(1+i)] <- x_test_melted[(1+(i-1)*293):(293+(i-1)*293),]
}

#y_train <- as.factor(y_train_py)
#y_test <- as.factor(y_test_py)
y_train <- rep(ifelse(y_train_py == 'positivity', 1, 0))#,28)
y_test <- rep(ifelse(y_test_py == 'positivity', 1, 0))#,28)



######### Transformada de Fourier

# CANAL 1
train_fft_1 <- fft(x_train$V2) 
train_fft_1 <- as.numeric(train_fft_1)
dim(train_fft_1) <- c(268, 896)

# CANAL 2
train_fft_2 <- fft(x_train$V3) 
train_fft_2 <- as.numeric(train_fft_2)
dim(train_fft_2) <- c(268, 896)

# CANAL 3
train_fft_3 <- fft(x_train$V4) 
train_fft_3 <- as.numeric(train_fft_3)
dim(train_fft_3) <- c(268, 896)

# CANAL 4
train_fft_4 <- fft(x_train$V5) 
train_fft_4 <- as.numeric(train_fft_4)
dim(train_fft_4) <- c(268, 896)

# CANAL 5
train_fft_5 <- fft(x_train$V6) 
train_fft_5 <- as.numeric(train_fft_5)
dim(train_fft_5) <- c(268, 896)

# CANAL 6
train_fft_6 <- fft(x_train$V6) 
train_fft_6 <- as.numeric(train_fft_6)
dim(train_fft_6) <- c(268, 896)


x_train_fft <- data.frame(id = 1:268)

x_train_fft[,(2)] <- train_fft_1[(1):(268),]
x_train_fft[,(3)] <- train_fft_2[(1):(268),]
x_train_fft[,(4)] <- train_fft_3[(1):(268),]
x_train_fft[,(5)] <- train_fft_4[(1):(268),]
x_train_fft[,(6)] <- train_fft_5[(1):(268),]
x_train_fft[,(7)] <- train_fft_6[(1):(268),]


################### FDA

#Transforma duas vari?veis em Functional Data
x.v2.fdata <- fdata(x_train_fft$V2)
x.v3.fdata <- fdata(x_train_fft$V3)
x.v4.fdata <- fdata(x_train_fft$V4)
x.v5.fdata <- fdata(x_train_fft$V5)
x.v6.fdata <- fdata(x_train_fft$V6)
x.v7.fdata <- fdata(x_train_fft$V7)




#Cria as bases Spline/Fourier
ldata.train=list("df"=as.data.frame(y_train),"x1"=x.v2.fdata,"x2" =x.v3.fdata, "x3" =x.v4.fdata, "x4" =x.v5.fdata, "x5" =x.v6.fdata, "x5" =x.v7.fdata)  
basis.x1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:8); basis.x2=create.fdata.basis(x.v3.fdata,type.basis="bspline", l=1:8) 
basis.x3=create.fdata.basis(x.v4.fdata,type.basis="bspline", l=1:8); basis.x4=create.fdata.basis(x.v5.fdata,type.basis="bspline", l=1:8) 
basis.x5=create.fdata.basis(x.v6.fdata,type.basis="bspline", l=1:8); basis.x6=create.fdata.basis(x.v7.fdata,type.basis="bspline", l=1:8) 

#basis.b1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:4); basis.b2=create.fdata.basis(x.v3.fdata,type.basis="bspline", l=1:4)   
basis.b1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:8); basis.b2=create.fdata.basis(x.v3.fdata,type.basis="bspline", l=1:8)
basis.b3=create.fdata.basis(x.v4.fdata,type.basis="bspline", l=1:8); basis.b4=create.fdata.basis(x.v5.fdata,type.basis="bspline", l=1:8)
basis.b5=create.fdata.basis(x.v6.fdata,type.basis="bspline", l=1:8); basis.b6=create.fdata.basis(x.v7.fdata,type.basis="bspline", l=1:8)


basis.x=list("x1"=basis.x1, "x2"=basis.x2, "x3"=basis.x3, "x4"=basis.x4, "x5"=basis.x5, "x6"=basis.x6)                                                   
basis.b=list("x1"=basis.b1, "x2"=basis.b2, "x3"=basis.b3, "x4"=basis.b4, "x5"=basis.b5, "x6"=basis.b6 )      

#Roda o GLM com a bas escolhida
res.basis=fregre.glm(y_train~x1 + x2 + x3 + x4 + x5,ldata.train,family=binomial(),basis.x=basis.x,basis.b=basis.b, CV=TRUE)    
#res.basis=fregre.gsam(y_train~s(x1)+s(x2)+s(x3)+s(x4) + s(x5),ldata.train,family=binomial,basis.x=basis.x,basis.b=basis.b) 
summary(res.basis)

table(y_train, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1)) #equivalente: table(y_train, ifelse(fitted(res.basis) < 0.5, 0, 1))
t_train <- table(y_train, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1))
t_train %>% diag() %>% sum () / t_train %>% sum()


roc_train <- roc(response = y_train, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1))
plotaroc(roc_train, titulo = "Curva ROC")


plot(x.v2.fdata, main = '')
plot(mean(x.v2.fdata), main = '')

plot(x.v3.fdata, main = '')
plot(mean(x.v3.fdata), main = '')

plot(x.v4.fdata, main = '')
plot(mean(x.v4.fdata), main = '')

plot(x.v5.fdata, main = '')
plot(mean(x.v5.fdata), main = '')

plot(x.v6.fdata, main = '')
plot(mean(x.v6.fdata), main = '')

plot(x.v7.fdata, main = '')
plot(mean(x.v7.fdata), main = '')

############ Base de teste #################
######### Transformada de Fourier

# CANAL 1
test_fft_1 <- fft(x_test$V2) 
test_fft_1 <- as.numeric(test_fft_1)
dim(test_fft_1) <- c(293, 896)

# CANAL 2
test_fft_2 <- fft(x_test$V3) 
test_fft_2 <- as.numeric(test_fft_2)
dim(test_fft_2) <- c(293, 896)

# CANAL 3
test_fft_3 <- fft(x_test$V4) 
test_fft_3 <- as.numeric(test_fft_3)
dim(test_fft_3) <- c(293, 896)

# CANAL 4
test_fft_4 <- fft(x_test$V5) 
test_fft_4 <- as.numeric(test_fft_4)
dim(test_fft_4) <- c(293, 896)

# CANAL 5
test_fft_5 <- fft(x_test$V6) 
test_fft_5 <- as.numeric(test_fft_5)
dim(test_fft_5) <- c(293, 896)

# CANAL 6
test_fft_6 <- fft(x_test$V6) 
test_fft_6 <- as.numeric(test_fft_6)
dim(test_fft_6) <- c(293, 896)


x_test_fft <- data.frame(id = 1:293)

x_test_fft[,(2)] <- test_fft_1[(1):(293),]
x_test_fft[,(3)] <- test_fft_2[(1):(293),]
x_test_fft[,(4)] <- test_fft_3[(1):(293),]
x_test_fft[,(5)] <- test_fft_4[(1):(293),]
x_test_fft[,(6)] <- test_fft_5[(1):(293),]
x_test_fft[,(7)] <- test_fft_6[(1):(293),]


################### FDA

#Transforma duas vari?veis em Functional Data
x.v2.fdata_test <- fdata(x_test_fft$V2)
x.v3.fdata_test <- fdata(x_test_fft$V3)
x.v4.fdata_test <- fdata(x_test_fft$V4)
x.v5.fdata_test <- fdata(x_test_fft$V5)
x.v6.fdata_test <- fdata(x_test_fft$V6)
x.v7.fdata_test <- fdata(x_test_fft$V7)



plot(mean(x.v2.fdata_test), main = '')

plot(mean(x.v3.fdata_test), main = '')

plot(mean(x.v4.fdata_test), main = '')

plot(mean(x.v5.fdata_test), main = '')

plot(mean(x.v6.fdata_test), main = '')

plot(mean(x.v7.fdata_test), main = '')



#Avalia performance (acuracia) nas bases de treino e teste
ldata.test=list("df"=as.data.frame(y_test),"x1"=x.v2.fdata_test,"x2" =x.v3.fdata_test, "x3" =x.v4.fdata_test, "x4" =x.v5.fdata_test, "x5" =x.v6.fdata_test, "x6" =x.v7.fdata_test) 


table(y_test, ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1))
t_test <- table(y_test, ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1))

t_test  %>% diag() %>% sum () / t_test  %>% sum()


predict_test <- ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1)
predict_test_x <- data.frame(predict = predict_test, y_test = y_test)

predict_test_x$predict <- as.factor(predict_test_x$predict)
predict_test_x$y_test <- as.factor(predict_test_x$y_test)

confusionMatrix(predict_test_x$predict, predict_test_x$y_test)
