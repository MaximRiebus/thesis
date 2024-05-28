import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time
from urllib.parse import urljoin, urlparse
from selenium.common.exceptions import TimeoutException

# Space inladen
nlp = spacy.load("en_core_web_md")

# Sustainability keywords opslaan
KEYWORDS = [
    'sustainability', 'about us', 'mission', 'values', 'purpose', 'corporate responsibility'
]

# Keywords omvormen naar vectoren
keyword_vectors = np.array([nlp(keyword).vector for keyword in KEYWORDS])

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_relevant_sentences(content, threshold=0.2):
    doc = nlp(content)
    relevant_sentences = []
    for sentence in doc.sents:
        sentence_vector = sentence.vector
        if sentence_vector.any():  
            similarities = cosine_similarity([sentence_vector], keyword_vectors)
            if np.max(similarities) > threshold:
                relevant_sentences.append(sentence.text)
    return " ".join(relevant_sentences)

def summarize_content(content, max_tokens=5000):
    if len(content.split()) <= max_tokens:
        return content
    sentences = content.split('. ')
    summary = []
    current_count = 0
    for sentence in sentences:
        tokens = len(sentence.split())
        if current_count + tokens > max_tokens:
            break
        summary.append(sentence)
        current_count += tokens
    return '. '.join(summary)

def filter_links_by_keyword(soup, base_url, max_links=3):
    relevant_links = []
    link_scores = []

    for link in soup.find_all('a', href=True):
        link_text = link.text.lower()
        link_vector = nlp(link_text).vector
        if link_vector.any():  
            similarities = cosine_similarity([link_vector], keyword_vectors)
            max_similarity = np.max(similarities)
            href = link['href']
            
            # URL joinen
            full_url = urljoin(base_url, href)

            # Dubbels eruit halen indien nodig
            url_parts = urlparse(full_url)
            path = url_parts.path
            unique_path_segments = []
            seen = set()

            # Splitten van paden
            for segment in path.split('/'):
                if segment not in seen:
                    unique_path_segments.append(segment)
                    seen.add(segment)

            # Paden reconstrueren
            normalized_path = '/'.join(unique_path_segments)
            full_url = full_url.replace(path, normalized_path)

            link_scores.append((full_url, max_similarity))
    
    # URLs sorteren op basis van meest relevant
    top_links = sorted(link_scores, key=lambda x: x[1], reverse=True)[:max_links]
    return [link[0] for link in top_links]

def filter_links_for_industry(soup, base_url, max_links=2):
    industry_keywords = ["about us", "about", "who we are", "company", "profile"]
    industry_keyword_vectors = np.array([nlp(keyword).vector for keyword in industry_keywords])
    relevant_links = [base_url]  
    link_scores = []

    for link in soup.find_all('a', href=True):
        link_text = link.text.lower()
        link_vector = nlp(link_text).vector
        if link_vector.any():  
            similarities = cosine_similarity([link_vector], industry_keyword_vectors)
            max_similarity = np.max(similarities)
            if max_similarity > 0.2:  
                href = link['href']
                full_url = urljoin(base_url, href)
                link_scores.append((full_url, max_similarity))
    
    # URLs sorteren op basis van meest relevant
    top_links = sorted(link_scores, key=lambda x: x[1], reverse=True)[:max_links]

    # Twee meest relevante URLs toevoegen
    for link in top_links:
        if link[0] != base_url:
            relevant_links.append(link[0])
            if len(relevant_links) - 1 == max_links:  
                break

    return relevant_links

def scrape_with_selenium(urls):
    options = Options()
    options.headless = True  
    scraped_content = {}
    for url in urls:
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            driver.set_page_load_timeout(20)  
            
            try:
                driver.get(url)
            except TimeoutException:
                logging.error(f"Timeout exceeded while loading {url}")
                scraped_content[url] = ""  
                continue 
            
            # Extra buffer
            time.sleep(3)

            # Scrollen
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3) 
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # Cookies en pop-ups proberen wegdoen indien nodig
            try:
                driver.find_element(By.XPATH, '//button[contains(text(), "Accept") or contains(text(), "agree")]').click()
            except Exception as e:
                logging.info("No accept button found or needed:", e)

            content = driver.find_element(By.TAG_NAME, 'main').text 
            scraped_content[url] = content

        except Exception as e:
            logging.error(f"Error scraping {url} with Selenium: {e}")
            scraped_content[url] = ""
        finally:
            driver.quit()
    return scraped_content

def scrape_and_combine_text(links):
    combined_text = ""
    for link in links:
        try:
            response = requests.get(link, timeout=10)
            if response.status_code == 200:
                link_soup = BeautifulSoup(response.content, 'html.parser')
                paragraphs = [p.get_text(separator=' ', strip=True) for p in link_soup.find_all('p')]
                page_text = " ".join(paragraphs)
                if page_text:
                    combined_text += page_text + " "
                if len(page_text.split()) < 50:
                    logging.info(f"Using Selenium for additional scraping due to insufficient content from {link}")
                    selenium_text = scrape_with_selenium([link])  
                    combined_text += " ".join(selenium_text.values()) + " "
            else:
                logging.error(f"Failed to scrape {link}: HTTP {response.status_code}")
        except requests.Timeout:
            logging.error(f"Timeout occurred when scraping {link}")
        except requests.RequestException as e:
            logging.error(f"Failed to scrape {link}: {e}")
    
    filtered_text = filter_relevant_sentences(combined_text)
    summarized_text = summarize_content(filtered_text)
    return summarized_text.strip()

def process_websites(file_path, output_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    df = pd.read_csv(file_path)
    results = []

    for index, row in df.iterrows():
        url = row['url']
        logging.info(f"Processing URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                filtered_links = filter_links_by_keyword(soup, url)
                combined_text = ""
                for link in filtered_links:
                    try:
                        link_response = requests.get(link, timeout=10)
                        if link_response.status_code == 200:
                            link_soup = BeautifulSoup(link_response.content, 'html.parser')
                            paragraphs = [p.get_text(separator=' ', strip=True) for p in link_soup.find_all('p')]
                            page_text = " ".join(paragraphs)
                            if page_text:
                                combined_text += page_text + " "
                    except requests.Timeout:
                        logging.error(f"Timeout occurred when scraping {link}")
                    except requests.RequestException as e:
                        logging.error(f"Error scraping {link}: {e}")

                if not combined_text or len(combined_text.split()) < 50:
                    logging.info("Insufficient content from BeautifulSoup, using Selenium scraper.")
                    selenium_content = scrape_with_selenium(filtered_links)
                    combined_text = " ".join(selenium_content.values())

                industry_links = filter_links_for_industry(soup, url, max_links=2)
                industry_content = scrape_and_combine_text(industry_links)

                result = {
                    'url': url,
                    'sustainability_relevant_urls': ';'.join(filtered_links),
                    'industry_relevant_urls': ';'.join(industry_links),
                    'scraped_text_for_sustainability': combined_text,
                    'scraped_text_for_industry': industry_content
                }
                results.append(result)
            else:
                logging.error(f"Failed to access {url}: HTTP {response.status_code}")
        except requests.Timeout:
            logging.error(f"Timeout occurred when accessing {url}")
        except requests.RequestException as e:
            logging.error(f"Network error processing {url}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    process_websites("websites_to_scrape.csv", "scraped_data.csv")