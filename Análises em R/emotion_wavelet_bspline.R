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

set.seed(10)
nrows <- nrow(data)
index <- sample(1:nrows, 0.7 * nrows)	# shuffle and divide
train_model <- data[index,]			       
test_model <- data[-index,] 


train_wt <- dwt(train_model$V2, filter="la8", boundary="periodic", fast=TRUE) # Transformada de Wavalet

#train_wt <- dwt(train_model$V2, filter="la8", boundary="reflection", fast=TRUE) # Transformada de Wavalet

x_train_melted_wt <- as.array(train_wt@series)

dim(x_train_melted_wt) <- c(128, 384)

x_train_wt <- data.frame(id = 1:128)

x_train_wt[,(2)] <- x_train_melted_wt[1:128,]

train_wt <- data.frame(train_model$Y, x_train_wt)


############ Base de Spline #############


x.v2.fdata <- fdata(train_wt$V2)

ldata.train=list("df"=as.data.frame(train_wt$train_model.Y),"x1"=x.v2.fdata) 
basis.x1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:8)
basis.b1=create.fdata.basis(x.v2.fdata,type.basis="bspline", l=1:8)


basis.x=list("x1"=basis.x1)                                                   
basis.b=list("x1"=basis.b1)    



x.v2.fdata_v2 <- fdata(train_model$V2)
plot(x.v2.fdata_v2)
plot(mean(x.v2.fdata), main = '')
plot(basis.x$x1)
plot(basis.b$x1)

set.seed(11)

res.basis=fregre.glm(train_wt$train_model.Y~x1,ldata.train,family=binomial,basis.x=basis.x,basis.b=basis.b) 
summary(res.basis)

table(train_wt$train_model.Y, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1)) #equivalente: table(y_train, ifelse(fitted(res.basis) < 0.5, 0, 1))
t_train <- table(train_wt$train_model.Y, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1))
t_train %>% diag() %>% sum () / t_train %>% sum()


plot(func.mean(fdata(train_wt$V2)))

roc_train <- roc(response = train_wt$train_model.Y, ifelse(predict(res.basis, newx = ldata.train) < 0.5, 0, 1))
plotaroc(roc_train, titulo = "Curva ROC")




########## test

#fft <- fft(test$V2) # Transformada de Fourier

test_wt <- dwt(test_model$V2, filter="la8", boundary="periodic", fast=TRUE) # Transformada de Wavalet

x_test_melted_wt <- as.array(test_wt@series)

dim(x_test_melted_wt) <- c(56, 384)

x_test_wt <- data.frame(id = 1:56)

x_test_wt[,(2)] <- x_test_melted_wt[1:56,]

test_wt <- data.frame(test_model$Y, x_test_wt)
####################


x.v2.fdata_test <- fdata(test_wt$V2)

ldata.test=list("df"=as.data.frame(test_wt$test_model.Y),"x1"=x.v2.fdata_test) 

t_test <- table(test_wt$test_model.Y, ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1))
t_test  %>% diag() %>% sum () / t_test  %>% sum()


predict_test <- ifelse(predict(res.basis, newx = ldata.test) < 0.5, 0, 1)
predict_test_x <- data.frame(predict = predict_test, y_test = test_wt$test_model.Y)

predict_test_x$predict <- as.factor(predict_test_x$predict)
predict_test_x$y_test <- as.factor(predict_test_x$y_test)

confusionMatrix(predict_test_x$predict, predict_test_x$y_test)

plot(func.mean(fdata(test_model$V2)))





























































########## EXAMPLE ##########
### Data Generation ###
# parameters for signal
Fs <- 1000 # 1000 Hz signal
s <- 3 # 3 seconds of data
t <- seq(0, s - 1/Fs, by = 1/Fs) # time sequence
n <- length(t) # number of data points
freqs <- c(1, 5, 10, 20) # frequencies
amp <- c(2, 1.5, 3, 1.75) # strengths (amplitudes)
phs <- c(0, pi/6, pi/4, pi/2) # phase shifts
# create data generating signals
mu <- rep(0, n)
for(j in 1:length(freqs)){
  mu <- mu + amp[j] * sin(2*pi*t*freqs[j] + phs[j])
}
set.seed(1) # set random seed
e <- rnorm(n) # Gaussian error
y <- mu + e # data = mean + error
### FFT of Noise-Free Data ###
# fft of noise-free data
ef <- eegfft(mu, Fs = Fs, upper = 40)
head(ef)
ef[ef$strength > 0.25,]
# plot frequency strength
par(mfrow = c(1,2))
plot(x = ef$frequency, y = ef$strength, t = "b",
     xlab = "Frequency (Hz)",
     ylab = expression("Strength (" * mu * "V)"),
     main = "FFT of Noise-Free Data")
# compare to data generating parameters
cbind(amp, ef$strength[ef$strength > 0.25])
cbind(phs - pi/2, ef$phase[ef$strength > 0.25])
### FFT of Noisy Data ###
# fft of noisy data
ef <- eegfft(y, Fs = Fs, upper = 40)
head(ef)
ef[ef$strength > 0.25,]
# plot frequency strength

plot(x = ef$frequency, y = ef$strength, t = "b",
     xlab = "Frequency (Hz)",
     ylab = expression("Strength (" * mu * "V)"),
     main = "FFT of Noisy Data")
# compare to data generating parameters
cbind(amp, ef$strength[ef$strength > 0.25])
cbind(phs - pi/2, ef$phase[ef$strength > 0.25])