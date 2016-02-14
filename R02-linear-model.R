prudential <- read.csv('~/my-sandbox/kaggle/151202-prudential/input/train.csv')
summary(prudential)
features <- subset(prudential, select = -c(Id, Response, Product_Info_2))

impute_na <- function(df)
{
  for(i in 1:ncol(df))
  {
    if(is.numeric(df[,i]))
    {
      df[is.na(df[,i]),i] <- median(df[!is.na(df[,i]),i])
    }
  }
  df
}

dat <- impute_na(features)


library(MASS)
lm01 = lm.ridge(prudential$Response ~ ., data = dat, lambda = 0.5)

lm02y = lm.ridge(prudential$Response ~ . -BMI, data = dat, lambda = 0.5)
lm02x = lm.ridge(BMI ~ . -BMI, data = dat, lambda =)
qplot(x = lm02x$coef, y = lm02y$coef, xlim = c(-0.0025,0.0025))

lm03y = lm.ridge(prudential$Response ~ . -Ins_Age, data = dat, lambda = 0.5)
lm03x = lm.ridge(Ins_Age ~ . -Ins_Age, data = dat, lambda =)
qplot(x = lm03x$coef, y = lm03y$coef)