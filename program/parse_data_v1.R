pkgs <- c('jsonlite','dplyr','ggplot2')
sapply(pkgs, require, character.only = TRUE)


options(stringsAsFactors = FALSE)

json_files <- list.files('raw')

var_names <- c('name','address','overall_rating','price_level','review_author','review_time',
               'review_relative_time','review_rating','review_text')
review_data <- data.frame(matrix(NA,nrow=0,ncol=length(var_names)))
names(review_data) <- var_names

tic = Sys.time()  
for (i in 1:length(json_files)){
  
  data <- fromJSON(paste0('raw/', json_files[i]))
  
  if (!is.null(data$result$reviews)){
    
    review_data <- rbind(review_data,
                         data.frame(name = data$result$name,
                                    address = data$result$formatted_address,
                                    overall_rating = ifelse(is.null(data$result$rating),NA,data$result$rating),
                                    price_level = ifelse(is.null(data$result$price_level),NA,data$result$price_level),
                                    review_author = data$result$reviews$author_name,
                                    review_time = data$result$reviews$time,
                                    review_relative_time = data$result$reviews$relative_time_description,
                                    review_rating = data$result$reviews$rating,
                                    review_text = data$result$reviews$text)
    )
  }
}
Sys.time() - tic

write.csv(review_data,'data/review_data_v1.csv',row.names=FALSE)
