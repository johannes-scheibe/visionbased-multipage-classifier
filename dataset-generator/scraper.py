from typing import List
from bs4 import BeautifulSoup
from time import sleep
from selenium.webdriver import Firefox
from selenium.webdriver.common.by import By

def scrape_url(url, num_pages) -> List[dict]:
    driver = Firefox(executable_path="firefox-driver/geckodriver")
    driver.get(url)

    ids = []
    docs:List[dict] = []

    p = 0
    while p < num_pages:
        sleep(2)
        html_doc = driver.execute_script("return document.getElementsByTagName('html')[0].innerHTML")
        
        page_docs = extract_content(html_doc)
    
        for d in page_docs:
            if d["id"] in ids:
                print(f"WARNING: Duplicate doc id found. This could be caused by the sleep timer")
                driver.close()
            ids.append(d["id"])
            
        docs.extend(page_docs)

        # Move to next page
        page_button = driver.find_element(By.ID, "header-page-forward")
        driver.execute_script("arguments[0].click();", page_button)
    
        p+=1
    driver.close()

    return docs
    

def extract_content(html):
    soup = BeautifulSoup(html, 'html.parser')

    entries = soup.find_all("div", attrs={"class": "result-entry"})
    docs = []
    for entry in entries:
        doc = {}

        doc_info = entry.find_all("div", attrs={"class": "result-datum"})
        for item in doc_info:
            key = item.find("dt").text.split(":")[0][:-1].lower().replace(" ", "_")
            value= item.find("dd").text

            doc[key] = value
        
        doc["pages"] = int(entry.find("div", attrs={"class": "result-icon-page-num"}).text.split(" ")[0])

        download_options = entry.find_all("a", attrs={'class': 'download-option'})
        for a in download_options:
            url:str = a.get("href")
            if url and url.endswith('pdf'):
                doc["download_url"] = url

        docs.append(doc)

    return docs
