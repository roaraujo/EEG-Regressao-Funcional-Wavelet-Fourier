require(reticulate)
require(dplyr)
library(refund)
library(mgcv)
library(pROC)
library(caret)
require(fda.usc)
library(pROC)
library(caret)
library(lattice)


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




data("emotion", package = "FDboost")
str(emotion)


length(emotion$EEG)



x_melted <- as.array(emotion$EEG)


dim(x_melted) <- c(184, 384)

X <- data.frame(id = 1:184)

X[,(2)] <- x_melted[1:184,]


Y <- rep(ifelse(emotion$game_outcome == 'gain', 1, 0))#,28) 
#Y <- rep(ifelse(emotion$power == 'high', 1, 0))#,28) 
#Y <- rep(ifelse(emotion$control == 'high', 1, 0))#,28) 

data <- data.frame(Y, X)

set.seed(11)
nrows <- nrow(data)
index <- sample(1:nrows, 0.7 * nrows)	# shuffle and divide
train_model <- data[index,]			       
test_model <- data[-index,] 


######### Transformada de Fourier

train_fft <- fft(train_model$V2) 
train_fft <- as.numeric(train_fft)

dim(train_fft) <- c(128, 384)

x_train_fft <- data.frame(id = 1:128)

x_train_fft[,(2)] <- train_fft[(1):(128),]


train_fft_dt <- data.frame(train_model$Y, x_train_fft)

############ Base de Spline #############

x.v2.fdata <- fdata(train_fft_dt$V2)

ldata.train=list("df"=as.data.frame(train_fft_dt$train_model.Y),"x1"=x.v2.fdata) 
basis.x1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:4)
basis.b1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:4)


basis.x=list("x1"=basis.x1)                                                   
basis.b=list("x1"=basis.b1)    



plot(x.v2.fdata)
plot(mean(x.v2.fdata))
plot(basis.x$x1)
plot(basis.b$x1)

set.seed(12)
res.basis=fregre.glm(train_fft_dt$train_model.Y ~ x1,ldata.train,family=binomial,basis.x=basis.x,basis.b=basis.b) 
summary(res.basis)

table(train_fft_dt$train_model.Y, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1)) #equivalente: table(y_train, ifelse(fitted(res.basis) < 0.5, 0, 1))
t_train <- table(train_fft_dt$train_model.Y, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1))
t_train %>% diag() %>% sum () / t_train %>% sum()


roc_train <- roc(response = train_fft_dt$train_model.Y, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1))
plotaroc(roc_train, titulo = "Curva ROC")




########## test

######### Transformada de Fourier

test_fft <- fft(test_model$V2) 
test_fft <- as.numeric(test_fft)

dim(test_fft) <- c(56, 384)

x_test_fft <- data.frame(id = 1:56)

x_test_fft[,(2)] <- test_fft[(1):(56),]


test_fft_dt <- data.frame(test_model$Y, x_test_fft)


####################


x.v2.fdata_test <- fdata(test_fft_dt$V2)

ldata.test=list("df"=as.data.frame(test_fft_dt$test_model.Y),"x1"=x.v2.fdata_test) 

t_test <- table(test_fft_dt$test_model.Y, ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1))
t_test  %>% diag() %>% sum () / t_test  %>% sum()


predict_test <- ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1)
predict_test_x <- data.frame(predict = predict_test, y_test = test_fft_dt$test_model.Y)

predict_test_x$predict <- as.factor(predict_test_x$predict)
predict_test_x$y_test <- as.factor(predict_test_x$y_test)

confusionMatrix(predict_test_x$predict, predict_test_x$y_test)

plot(func.mean(fdata(test_fft_dt$V2)))


