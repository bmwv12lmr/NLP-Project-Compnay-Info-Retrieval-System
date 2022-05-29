# CMPE-257 Final Exam
Allen Wu 015292667 yanshiun.wu@sjsu.edu

## Project Demo
![Graph](https://github.com/bmwv12lmr/CMPE-257_Final/blob/main/IMG/giftpack.jpg)

## Requirements:
1. Get access to crunchbase API  
    Use BeautifulSoup to scrape crunchbase instead of crunchbase API.
    Due to unavailability of crunchbase API since crunchbase didn't asssign the API Key for me after filing the request for a long tiem.
  

2. accept the name of a company  
    Please type the company name in variable "COMPANY_NAME" below  


3. search for the company on crunchbase  
    Use function "get_url" to get the page of company on crunchbase


4. search for company in the news  
    Use function "get_google_news" can perform news searching and return news list


5. POS and NER for building a graph of relationships  
    Use POS to find the cos sim form company's webpage and use NER to find the entities from company and keymans' news to build relationship in grape


6. lookup the main people (NER) from crunchbase using search api (e.g., google search api)  
    Use NER to find the entities from keymans' Google News Search


7. get top topics in the news about the company  
    Use BERTopic to find the topic of company news and add it to the graph


8. show graph using networkx api in python  
    Use networkx API to build graph in function plot_company_info and save the graph in IMG folder with COMPANY_NAME.jpg


9. deposit a text file containing research about that company  
    Use function output_company_info to save the info in RPT folder with COMPANY_NAME.txt

## Instruction
1. Install required packages
> pip install -r requirements.txt
2. Check Google Chrome version and download chromedriver for scraping
> Visit https://chromedriver.chromium.org/downloads
3. Enable scrape_mode in main.py to start scraping
> scrape_mode = True
4. Run main.py


5. If needed, please press verification button which showed when running chromedriver


6. After running, check IMG and RPT folders and find files with target company for result
> IMG/company_name.jpg is the Relation Graph  
> RPT/company_name.txt is the text report  

## NOTE
If you are using Windows, please rename chromedriver.exe to chromedriver

## Folders
* CRUNCHBASE stores analyzed Cruchbase data
* IMG stores generated relation graphs
* RPT stores generated text reports
* RSS stores news report with hashed timestamp
* WEB store stores scraped web data
