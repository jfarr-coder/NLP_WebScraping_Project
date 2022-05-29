# NLP_WebScraping_Project

## Purpose of the Project

This is a project for my CS 796 course, Natural Language Processing, provided by Old Dominion University and taught by [Dr. Vikas Ganjigunte Ashok](https://scholar.google.com/citations?user=Of8dNP0AAAAJ&hl=en).

Some news articles have incomplete headlines in the form of questions, vague points, or incomplete sentences.  This is done to entice the reader to click and view their page. Unfortunately, the reader would be wasting their time just to get an answer to the headline, which would sometime be at the end of the article.

Why not cut the time and create a tool that would show the answer JUST by hovering over an article's headline?

That was the goal I've chosen from a list of available projects from the course.  However, I could not create the tool without classifying the articles based on whether they had Complete (C) or Incomplete (I) headlines.

## Data

I manually gathered data from all types of news websites that covered a range of subjects, from politices to entertainment.  I had a spreadsheet for the Commplete articles and the Incomplete articles EACH.  The labels are below.

Complete
- Source (CNN, Yahoo, etc.)
- Heading (Heading on the site)
- C/I (Complete/Incomplete)
- Full Text (Article)
- Full Text Link (Link to the page)
- Date (Date the article was published)

Incomplete
- Source (CNN, Yahoo, etc.)
- Heading (Heading on the site)
- C/I (Complete/Incomplete)
- Full Text (Article)
- Full Text Link (Link to the page)
- Answer Portions (Portions of the full text containing the answer to the headline)
- Date (Date the article was published)

The data of both types of articles are stored in Comma Separated Value (CSV) spreadsheets.

## Technologies Used

The project was written in python3.  I used the [Pandas](https://pandas.pydata.org/) library to processe and organize the data.  I also used both the [NLTK(Natural Langualge Tool Kit)](https://www.nltk.org/) and [StanfordCoreNLP](https://stanfordnlp.github.io/CoreNLP/) libraries to implement POS(Part-Of-Speech) Tagging and Sentence Parsing. 

I ran the code, and gathered the results, in an Anaconda environment.  

## How to Launch

While I ran the code in an Ubuntu Sub-system, you should be able to run it in any CLI you're comfortable with.  You do have to export the Anaconda environment first.

``conda env create --file NLP_Project.yml``

You must then enter the environment, by activating it using the command below.
 
 ``conda activate NLP_Project``
 
You would then have to have TWO instances of your CLI.  In the first instance, you need to go into the stanford-core-nlp-4.4.0 folder and run the command below.

``java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9002 -timeout 30000``

The CoreNLP library is a server written in Java.  When you are running the command above, you are activating the server on port 9002, which will run the commands needed to parse the data.  You will also notice that the python3 script is calling to the same port to get the requested results from said commands.

In the second instance, you can just run the python3 script in the parent directory, but make sure you are in the NLP_Project environment before you do.  Otherwise, the script will not work.  Keep an eye on both instances, and you will notice that the server is taking all the data and parsing them. 

``python3 NLP_Classfier.py``

## Powerpoint

Visuals are included in the PPT below.
https://docs.google.com/presentation/d/12DIzklfy5YZ8C0i7BR2ZHtDY71VR2go1ITxGwr7wnE4/edit?usp=sharing

## Progress

To be honest, I was only able to go as far as calculating the Term Frequencies(TF) of each POS tag, Parser tag, unigram, and bigram.  The TFs were all outputted into the results.csv file.  

As a result, you can do whatever you want with this repo and use the data however you want.  Best of Luck!
