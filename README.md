# CMPE-257 Final Exam
Allen Wu 015292667 yanshiun.wu@sjsu.edu

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

## NOTE
If you want to scrape data from CrunchBase, you MUST ENABLE variable "scrape_mode" to avoid blocked by crunchbase anti-bot tools

'''
