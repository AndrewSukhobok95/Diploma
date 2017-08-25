library(forecast)
library(openxlsx)
library(ggplot2)

file <- loadWorkbook('Data.xlsx')
sheets(file)

df <- read.xlsx(file, sheet="US_FederalFundsEffectiveR")
full_ts <- as.numeric(df[6:nrow(df), 2])
train <- full_ts[1:585]
test <- full_ts[586:length(full_ts)]

plot(train, type='l', col='blue')
plot(test, type='l', col='red')

model <- arima(train, order = c(2,1,1))
summary(model)
forec <- forecast.Arima(model, h = 146)
plot(forec)
plot(full_ts, type='l')
