import requests
from bs4 import BeautifulSoup
#https://realpython.com/beautiful-soup-web-scraper-python/
#https://www.geeksforgeeks.org/extract-all-the-urls-from-the-webpage-using-python/
URL = "https://www.cnn.com/politics"
page = requests.get(URL)

#def getURLS(argv):
    
if __name__=="__main__":
    soup = BeautifulSoup(page.text, 'html.parser')
    urls = []
    for link in soup.find_all('a'):
        print(link.get('href'))