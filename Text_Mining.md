Text Mining
================
Jared
November 27, 2018

Gutenberg contains multiple texts. We will examine some texts from a few famous authors.

``` r
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(tidytext)
library(gutenbergr)
library(tidyr)
library(stringr)
library(ggplot2)
library(tm)
```

    ## Loading required package: NLP

    ## 
    ## Attaching package: 'NLP'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     annotate

``` r
library(topicmodels)
library(SnowballC)
```

``` r
#Downloading the text
text = gutenberg_download(c(2147,2148,2149,2150,2151))
```

    ## Determining mirror for Project Gutenberg from http://www.gutenberg.org/robot/harvest

    ## Using mirror http://aleph.gutenberg.org

``` r
tidy_poe <- text %>%
  unnest_tokens(word, text) %>% 
  anti_join(stop_words)
```

    ## Joining, by = "word"

Top 10 word appearances

``` r
tidy_poe %>% 
count(word, sort = TRUE)
```

    ## # A tibble: 22,807 x 2
    ##    word       n
    ##    <chr>  <int>
    ##  1 time     591
    ##  2 found    567
    ##  3 length   386
    ##  4 day      382
    ##  5 eyes     359
    ##  6 head     349
    ##  7 night    328
    ##  8 water    309
    ##  9 left     287
    ## 10 feet     280
    ## # ... with 22,797 more rows

John Bunyan is one of the Christian English writer and Puritan teacher, the author of Christian allegory The Pilgrim's Progress. Gutenberg id 6046,6047,6048

``` r
bunyan_text <- gutenberg_download(c(6046,6047,6048))
```

Top ten words from Bunyan

``` r
tidy_bunyan <- bunyan_text %>%
  unnest_tokens(word, text) %>% 
  anti_join(stop_words)
```

    ## Joining, by = "word"

``` r
tidy_bunyan %>% 
count(word, sort = TRUE)
```

    ## # A tibble: 24,575 x 2
    ##    word       n
    ##    <chr>  <int>
    ##  1 god    18735
    ##  2 christ  9839
    ##  3 thou    8408
    ##  4 1       6266
    ##  5 thy     5825
    ##  6 lord    5561
    ##  7 2       5052
    ##  8 thee    4540
    ##  9 sin     4450
    ## 10 hath    4446
    ## # ... with 24,565 more rows

Using the AFINN sentiment dictionary, what is the sum sentiment score of the five documents from Poe.

``` r
get_sentiments('afinn')
```

    ## # A tibble: 2,476 x 2
    ##    word       score
    ##    <chr>      <int>
    ##  1 abandon       -2
    ##  2 abandoned     -2
    ##  3 abandons      -2
    ##  4 abducted      -2
    ##  5 abduction     -2
    ##  6 abductions    -2
    ##  7 abhor         -3
    ##  8 abhorred      -3
    ##  9 abhorrent     -3
    ## 10 abhors        -3
    ## # ... with 2,466 more rows

``` r
tidy_poe_sentiment <- tidy_poe %>%
                          inner_join(get_sentiments("afinn"), by = "word") %>%
                          summarise(sentiment = sum(score))
  
head(tidy_poe_sentiment)
```

    ## # A tibble: 1 x 1
    ##   sentiment
    ##       <int>
    ## 1     -1570

Using the AFINN sentiment dictionary, what is the sum sentiment score of the three documents from Bunyan.

``` r
tidy_bunyan_sentiment <- tidy_bunyan %>%
                          inner_join(get_sentiments("afinn"), by = "word") %>%
                          summarise(sentiment = sum(score))
  
head(tidy_bunyan_sentiment)
```

    ## # A tibble: 1 x 1
    ##   sentiment
    ##       <int>
    ## 1     25799

Topic Modeling
--------------

Latent Dirichlet allocation (LDA) is one of the most common algorithms for topic modeling. Here we will examine topic modeling for the texts: Gutenberg id: 932 - The Fall of the House of Usher, 98 - Tale of two Cities, 19337 - A Christmas Carol, 39452 - Pilgrim's progress.

``` r
# List of titles
titles <- c("The Fall of the House of Usher", "Tale of two Cities", "A Christmas Carol",
            "Pilgrim's progress")
books <- gutenberg_works(title %in% titles) %>%
  gutenberg_download(meta_fields = "title")
```

As pre-processing, we divide these into chapters, use tidytext's unnest\_tokens() to separate them into words, then remove stop\_words. We're treating every chapter as a separate "document", each with a name like Great Expectations\_1 or Pride and Prejudice\_11. (In other applications, each document might be one newspaper article, or one blog post).

``` r
# split into words
by_word <- books %>%
  unnest_tokens(word, text)

# find document-word counts
word_counts <- by_word %>%
  anti_join(stop_words) %>%
  count(title, word, sort = TRUE) %>%
  ungroup()
```

    ## Joining, by = "word"

``` r
word_counts
```

    ## # A tibble: 5,698 x 3
    ##    title             word          n
    ##    <chr>             <chr>     <int>
    ##  1 A Christmas Carol scrooge     327
    ##  2 A Christmas Carol christmas    96
    ##  3 A Christmas Carol ghost        91
    ##  4 A Christmas Carol spirit       86
    ##  5 A Christmas Carol time         68
    ##  6 A Christmas Carol cried        56
    ##  7 A Christmas Carol bob          49
    ##  8 A Christmas Carol scrooge's    48
    ##  9 A Christmas Carol door         45
    ## 10 A Christmas Carol hand         43
    ## # ... with 5,688 more rows

``` r
titles_dtm <- word_counts %>%
  cast_dtm(title, word, n)

titles_dtm
```

    ## <<DocumentTermMatrix (documents: 2, terms: 4986)>>
    ## Non-/sparse entries: 5698/4274
    ## Sparsity           : 43%
    ## Maximal term length: 16
    ## Weighting          : term frequency (tf)

``` r
titles_lda <- LDA(titles_dtm, k = 4, topn=10, control = list(seed = 1234))
titles_lda
```

    ## A LDA_VEM topic model with 4 topics.

``` r
chapter_topics <- tidy(titles_lda, matrix = "beta")

top_terms <- chapter_topics %>%
  group_by(topic) %>%
  top_n(5, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms
```

    ## # A tibble: 20 x 3
    ##    topic term         beta
    ##    <int> <chr>       <dbl>
    ##  1     1 mental    0.00877
    ##  2     1 mind      0.00801
    ##  3     1 power     0.00699
    ##  4     1 direction 0.00675
    ##  5     1 lay       0.00654
    ##  6     2 scrooge   0.0303 
    ##  7     2 christmas 0.0112 
    ##  8     2 spirit    0.00939
    ##  9     2 time      0.00807
    ## 10     2 door      0.00804
    ## 11     3 scrooge   0.0339 
    ## 12     3 cried     0.0102 
    ## 13     3 ghost     0.0101 
    ## 14     3 christmas 0.00755
    ## 15     3 spirit    0.00741
    ## 16     4 usher     0.00792
    ## 17     4 house     0.00586
    ## 18     4 character 0.00478
    ## 19     4 wild      0.00418
    ## 20     4 portion   0.00414

``` r
chapter_topics %>%
  filter(term == "heart")
```

    ## # A tibble: 4 x 3
    ##   topic term     beta
    ##   <int> <chr>   <dbl>
    ## 1     1 heart 0.00325
    ## 2     2 heart 0.00273
    ## 3     3 heart 0.00235
    ## 4     4 heart 0.00146

``` r
top_terms %>%
  mutate(term = reorder(term, beta)) %>%
  ggplot(aes(term, beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip()
```

![](Text_Mining_files/figure-markdown_github/unnamed-chunk-17-1.png)

``` r
chapters_gamma <- tidy(titles_lda, matrix = "gamma")
chapters_gamma
```

    ## # A tibble: 8 x 3
    ##   document                       topic      gamma
    ##   <chr>                          <int>      <dbl>
    ## 1 A Christmas Carol                  1 0.00000764
    ## 2 The Fall of the House of Usher     1 0.190     
    ## 3 A Christmas Carol                  2 0.512     
    ## 4 The Fall of the House of Usher     2 0.0000280 
    ## 5 A Christmas Carol                  3 0.488     
    ## 6 The Fall of the House of Usher     3 0.0000280 
    ## 7 A Christmas Carol                  4 0.00000764
    ## 8 The Fall of the House of Usher     4 0.810
