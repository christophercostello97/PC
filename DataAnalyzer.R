install.packages('caret', 'kernlab', 'syuzhet', 'tm', 'topicmodels')

NGramTokenizer <- function(x, n = 2) {
  trim <- function(x) gsub("(^\\s+|\\s+$)", "", x)
  terms <- strsplit(trim(x), split = "\\s+")[[1]]
  ngrams <- vector()
  if (length(terms) >= n) {
    for (i in n:length(terms)) {
      ngram <- paste(terms[(i-n+1):i], collapse = " ")
      ngrams <- c(ngrams, ngram)
    }
  }
  ngrams  
}

CustomTokenizer <- function(x) c(NGramTokenizer(x, 1), NGramTokenizer(x, 2), NGramTokenizer(x, 3), NGramTokenizer(x, 4))


data.path <- file.path(getwd(), "..", "Data", "Processed", "twitter_data_processed.csv")  # get path to file containing labeled Tweets
data <- read.csv(data.path, header = TRUE)  # read labeled Tweets into dataframe

library(tm)  # use Text Mining Package for manipulating raw text
data.corpus <- VCorpus(VectorSource(data$text), readerControl = list(language = "en"))  # read Tweet text into corpus
data.corpus <- tm_map(data.corpus, content_transformer(function(x) iconv(x, from = "latin1", to = "ASCII", sub = "")))  # convert characters to ASCII
data.corpus <- tm_map(data.corpus, content_transformer(gsub), pattern = "#\\w+", replacement = "")  # remove hashtags
data.corpus <- tm_map(data.corpus, content_transformer(gsub), pattern = "@\\w+", replacement = "")  # remove user handles
data.corpus <- tm_map(data.corpus, content_transformer(gsub), pattern = "(RT|via)((?:\\b\\W*@\\w+)+)", replacement = "")  # Remove Retweet information
data.corpus <- tm_map(data.corpus, content_transformer(gsub), pattern = "http[[:alnum:]]*", replacement = "")  # remove URLs
data.corpus <- tm_map(data.corpus, content_transformer(gsub), pattern = "[^a-zA-Z]+", replacement = " ")  # substitute punctuation with space
data.corpus <- tm_map(data.corpus, content_transformer(gsub), pattern = "[0-9]+", replacement = " ")  # substitute numbers with space
data.corpus <- tm_map(data.corpus, content_transformer(tolower))  # make characters lowercase
data.corpus <- tm_map(data.corpus, removeWords, c(stopwords("english")))  # remove English stopwords
data.corpus <- tm_map(data.corpus, stemDocument)  # stem words
data.corpus <- tm_map(data.corpus, stripWhitespace)  # remove spare whitespace

data.dtm <- DocumentTermMatrix(data.corpus, control = list(tokenize = CustomTokenizer))  # read corpus into document-term matrix (DTM) of unigrams, bigrams, trigrams and quadgrams
data.dtm <- removeSparseTerms(data.dtm, 0.999)  # remove .1% most sparse terms from DTM

library(topicmodels)
data.lda <- LDA(data.dtm, k = 20, control = list(seed = seed))  # classify data by 20 topics
data.lda.gamma <- as.data.frame(data.lda@gamma)  # get dataframe of topics per document

data.dtm <- weightTfIdf(data.dtm, normalize = FALSE)  # re-weigh terms by inverse document frequency (IDF)

data.row.sum <- apply(data.dtm, 1, sum)  # get sums of rows
data.empty.index <- as.numeric(data.dtm[data.row.sum == 0, ]$dimnames[1][[1]])  # index empty rows (documents without any discernable terms)
data <- data[-data.empty.index, ]  # remove indexed rows from dataframe
data.corpus <- data.corpus[-data.empty.index]  # remove indexed rows from corpus
data.dtm <- data.dtm[-data.empty.index, ]  # remove indexed rows from DTM

data.container <- as.data.frame(as.matrix(data.dtm))  # convert DTM into manipulable dataframe
data.container <- cbind.data.frame(data.container, data.lda.gamma)
levels(data$label) <- make.names(levels(data$label))  # convert label levels into syntactically valid names to be interpreted by model
library(syuzhet) 
data.container$SENTIMENT <- get_sentiment(data$text)
data.container$LABEL <- data$label  # attach labels to respective rows in DTM dataframe

set.seed(47)  # set seed to ensure reproducible results

library(caret)  # use Classification and Regression Training package to train machine learning models
data.train.index <- createDataPartition(data.container$LABEL, p = 0.8, list = FALSE)  # index sample of 80% of DTM dataframe for training data
data.train <- data.container[data.train.index, ]  # subset training data
data.test <- data.container[-data.train.index, ]  # subset testing data

model.svmLinear.control <- trainControl(method = "repeatedcv",  # cross-validate
                                        number = 10,  # tenfold cross-validation
                                        repeats = 3,  # repeat thrice and average results
                                        classProbs = TRUE,  # compute probabilities of classification accuracy
                                        summaryFunction = twoClassSummary)  # use area under curve to pick the best model
model.svmLinear.tune <- expand.grid(C = 10^(-1:2))  # try four values for cost
library(kernlab)
model.svmLinear <- train(LABEL ~ .,  # classify by label
                         data = data.train,  # use training data
                         method = "svmLinear",  # use linear SVM
                         preProcess = c("center", "scale"),  # standardize (center and scale) data
                         metric = "ROC",
                         trControl = model.svmLinear.control,
                         tuneGrid = model.svmLinear.tune)