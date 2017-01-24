# Please download the latest version of spark from 
# http://spark.apache.org/downloads.html

# Env Variables
if (nchar(Sys.getenv("SPARK_HOME")) < 1) {
  # Please change to location to your spark home
  Sys.setenv(SPARK_HOME = "/Users/krishna/spark-2.1.0-bin-hadoop2.7/")
}

# Load Spark R libraries
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))


# Data frame operations
df <- as.DataFrame(iris)
printSchema(df)
collect(describe(df))

head(df[,5],10)
df.filter <- filter(df, df$Sepal_Length > 5.7)
head(df.filter,10)


# Model Persistence
model.iris <- spark.glm(df, Sepal_Length ~ Sepal_Width + Species, family = "gaussian")
summary(model.iris)
modelPath <- tempfile(pattern = "MOOC/ml", fileext = ".tmp") 
write.ml(gaussianGLM, '~/tmp_model') 

# Load model
model2.iris <- read.ml('~/tmp_model')
summary(model2.iris)

# UDF
# lapply 
# (distributes the computations with Spark)
# Similar to doParallel

# Example 1
families <- c("gaussian", "poisson")
train <- function(family) {
  model <- glm(Sepal.Length ~ Sepal.Width + Species, iris, family = family)
  summary(model)
}

spark.lapply(families, train)


# Example 2
nu <- c(0.5,100)
train.svm <- function(nu) {
  library("e1071")
  model.svm <- svm(Species ~ Sepal.Width +Petal.Length + Petal.Width,
                   data=iris, nu=nu, type = 'C-classification', kernel = 'radial')
  y.hat <- predict(model.svm, iris)
  cm <- table(iris$Species, y.hat)
}

spark.lapply(nu, train.svm)

# gapply 
# (function to each partition of a SparkDataFrame)
result <- gapplyCollect(
  df,
  "Species",
  function(key, x) {
    y <- data.frame(key, max(x$Sepal_Length))
    colnames(y) <- c("Species", "max_sepal_length")
    y
  })
head(result)

