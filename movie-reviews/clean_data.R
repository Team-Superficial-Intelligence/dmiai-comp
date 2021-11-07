library(tidyverse)


review_text <- read_lines("./data/movies.txt.gz")

rscore <- str_subset(review_text, pattern = "^review/score")

rtext <- str_subset(review_text, pattern = "^review/text")

rm(review_text)

df <- data.frame(text = rtext, score = rscore)

rm(rscore)
rm(rtext)

head(df)
str_length("review/score: ")
df$score_cln <- substring(df$score, str_length("review/score: "))
df$text_cln <- substring(df$text, str_length("review/text: "))

df2 <- data.frame(text = df$text_cln, score = df$score_cln)

rm(df)

df2$score <- as.numeric(df2$score)

write_csv(df2, file.path("./data", "amazon.csv.gz"))

rm(df2)