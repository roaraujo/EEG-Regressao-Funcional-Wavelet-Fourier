require(reticulate)
require(dplyr)
library(refund)
library(mgcv)
library(pROC)
library(caret)
require(fda.usc)

#https://rdrr.io/github/fdboost/FDboost/man/emotion.html
#Gravações de EEG e EMG em um estudo computadorizado de jogos de azar
#Descrição
#Para analisar a relação funcional entre eletroencefalografia (EEG) e eletromiografia facial (EMG), Gentsch et al. (2014) registraram simultaneamente os sinais de EEG e EMG de 24 participantes enquanto eles jogavam uma tarefa de jogo computadorizada. O subconjunto fornecido contém observações agregadas de 23 participantes. As curvas foram calculadas em média sobre cada sujeito e cada uma das 8 configurações de estudo, resultando em 23 vezes 8 curvas.

#Formato
#Uma lista com as 10 variáveis a seguir.

#power
#variável de fator com níveis alto e baixo

#game_outcome
#fator variável com níveis de ganho e perda

#control
#variável de fator com níveis alto e baixo

#subject
#variável de fator com 23 níveis

#EEG
#matriz; Sinal EEG em formato amplo

#EMG
#matriz; Sinal EMG em formato amplo

#s
#pontos de tempo para a covariável funcional

#t
#pontos de tempo para a resposta funcional

#Detalhes
#O objetivo é explicar os potenciais no sinal EMG por configurações de estudo, bem como o sinal EEG (ver Ruegamer et al., 2018).

#Fonte
#Gentsch, K., Grandjean, D. e Scherer, KR (2014) Coerência explorada entre os componentes da emoção: evidências de potenciais relacionados a eventos e eletromiografia facial. Biological Psychology, 98, 70-81.

#Ruegamer D., Brockhaus, S., Gentsch K., Scherer, K., Greven, S. (2018). Impulsionando modelos históricos funcionais específicos de fatores para a detecção de sincronização em sinais bioelétricos. Journal of the Royal Statistical Society: Series C (Applied Statistics), 67, 621-642.

data("emotion", package = "FDboost")
str(emotion)



length(emotion$EEG)



x_train_melted <- as.array(emotion$EEG)


set.seed(23) 


dim(x_train_melted) <- c(184, 384)

x_train <- data.frame(id = 1:184)

x_train[,(2)] <- x_train_melted[1:184,]


y_train <- rep(ifelse(emotion$game_outcome == 'gain', 1, 0))#,28) 


train <- data.frame(y_train, x_train)


nrows <- nrow(train)
index <- sample(1:nrows, 0.7 * nrows)	# shuffle and divide
train_model <- train[index,]			       
test_model <- train[-index,] 



############ Base de Spline #############


x.v2.fdata <- fdata(train_model$V2)

ldata.train=list("df"=as.data.frame(train_model$y_train),"x1"=x.v2.fdata) 
basis.x1=create.fdata.basis(x.v2.fdata,type.basis="fourier", l=1:10)
basis.b1=create.fdata.basis(x.v2.fdata,type.basis="fourier", l=1:10)


basis.x=list("x1"=basis.x1)                                                   
basis.b=list("x1"=basis.b1)    


plot(x.v2.fdata)
plot(basis.x$x1)
plot(basis.b$x1)



res.basis=fregre.gsam(train_model$y_train~s(x1),ldata.train,family=binomial,basis.x=basis.x,basis.b=basis.b) 
summary(res.basis)

table(train_model$y_train, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1)) #equivalente: table(y_train, ifelse(fitted(res.basis) < 0.5, 0, 1))
t_train <- table(train_model$y_train, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1))
t_train %>% diag() %>% sum () / t_train %>% sum()


plot(func.mean(fdata(train_model$V2)))






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



library(pROC)
library(lattice)
roc_train <- roc(response = train_model$y_train, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1))
plotaroc(roc_train, titulo = "Curva ROC")




########## test


library(caret)

x.v2.fdata_test <- fdata(test_model$V2)

ldata.test=list("df"=as.data.frame(test_model$y_train),"x1"=x.v2.fdata_test) 

t_test <- table(test_model$y_train, ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1))
t_test  %>% diag() %>% sum () / t_test  %>% sum()


predict_test <- ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1)
predict_test_x <- data.frame(predict = predict_test, y_test = test_model$y_train)

predict_test_x$predict <- as.factor(predict_test_x$predict)
predict_test_x$y_test <- as.factor(predict_test_x$y_test)

confusionMatrix(predict_test_x$predict, predict_test_x$y_test)

plot(func.mean(fdata(test_model$V2)))








library(wavelets)
#library(wavethresh)
#wave <- wp(x_train_melted)
#wt <- dwt(train_model$V2, filter="la8", boundary="periodic", fast=TRUE)
#wt@series








#data(phoneme)
#ldata2_t=list("df"=data.frame(glearn=phoneme$classlearn),"x"=phoneme$learn)


## Machine Learning

#ldata_svm=list("df"=data.frame(y=ldata.train[["df"]]),"x"=ldata.train[["x1"]])
#require(e1071)
#res.svm=classif.svm(y ~ x, data = ldata.train)
# require nnet package
#res.nnet=classif.nnet(glearn~x,data=ldata,trace=FALSE)
# require rpart package
#res.rpart=classif.rpart(glearn~x,data=ldata)
#round(mean(res.svm$prob.classification),3)


