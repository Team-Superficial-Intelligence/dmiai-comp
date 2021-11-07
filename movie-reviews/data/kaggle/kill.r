library(tidyverse)

df <- read_csv("data/kaggle/train.csv")

df$zero <- 0
df$one <- 0
df$two <- 0
df$three <- 0
df$four <- 0


df[df$label == 0,]$zero <- 1
df[df$label == 1,]$one <- 1
df[df$label == 2,]$two <- 1
df[df$label == 3,]$three <- 1
df[df$label == 4,]$four <- 1

write_csv(df, "train_fixed.csv")
