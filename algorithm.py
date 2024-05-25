import requests
from bs4 import BeautifulSoup
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time
from collections import Counter
import statistics
import os
import re
from urllib.parse import urljoin, urlparse
from selenium.common.exceptions import TimeoutException
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score

# Spacy inladen
nlp = spacy.load("en_core_web_md")

# Sustainability kernwoorden opslaan in lijst
KEYWORDS = [
    'sustainability', 'about us', 'mission', 'values', 'purpose', 'corporate responsibility'
]

# Kernwoorden omvormen naar vecotren
keyword_vectors = np.array([nlp(keyword).vector for keyword in KEYWORDS])

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API key
openai.api_key = "secret"

# Functie die relevante zinnen ivm duurzaamheid uit tekst verzamelt
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

# Functie die tekst samenvat tot maximaal 5000 tokens omdat dit de max is voor de OpenAI API
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

# Functie die maximaal drie relevante webpagina's teruggeeft op basis van afstand van relevante vectoren (duurzaamheid)
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
            
            # url joinen
            full_url = urljoin(base_url, href)

            # duplicate paden eruit halen als het nodig is
            url_parts = urlparse(full_url)
            path = url_parts.path
            unique_path_segments = []
            seen = set()
            # paden van de url opsplitsen en combineren tot een juist geheel
            for segment in path.split('/'):
                if segment not in seen:
                    unique_path_segments.append(segment)
                    seen.add(segment)
            # paden reconstrueren
            normalized_path = '/'.join(unique_path_segments)
            full_url = full_url.replace(path, normalized_path)

            link_scores.append((full_url, max_similarity))
    
    # links sorteren en de drie meest relevante eruit kiezen
    top_links = sorted(link_scores, key=lambda x: x[1], reverse=True)[:max_links]
    return [link[0] for link in top_links]

# Functie die maximaal twee relevante webpagina's + home pagina teruggeeft op basis van afstand van relevante vectoren (industrie)
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
    
    # Sort links by highest similarity score and select the top links based on max_links
    top_links = sorted(link_scores, key=lambda x: x[1], reverse=True)[:max_links]

    # Append only the most relevant links if they're different from the base URL
    for link in top_links:
        if link[0] != base_url:
            relevant_links.append(link[0])
            if len(relevant_links) - 1 == max_links:  
                break

    return relevant_links

# Functie die dynamische webpagina's scrapete
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
            
            # indien nodig extra buffer toevoegen voor dynamische content
            time.sleep(3)

            # scrollen op de pagina
            last_height = driver.execute_script("return document.body.scrollHeight")
            while True:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(3) 
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            # cookies en pop-ups weg proberen halen indien nodig
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

# Functie die teskten van relevante webpagina's verzamelt en combineert (duurzaamheid en industrie)
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
                # als de inhoud van de gevonden tekst te weinig is, wordt hier ook gebruik gemaakt van Selenium
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
    
    # teksten filteren en samenvatten 
    filtered_text = filter_relevant_sentences(combined_text)
    summarized_text = summarize_content(filtered_text)
    return summarized_text.strip()

# OpenAI API call voor duurzaamheid (met context)
def analyze_sustainability(content):
    max_content_length = 4999
    if len(content) > max_content_length:
        content = content[:max_content_length]

    prompt = f"""
    Analyze the following text to determine the company's commitment to sustainability based on the provided descriptions:

    Text: {content}

    Sustainability scale:
    1. No importance to sustainability: No mention of sustainability initiatives, policies or objectives on the website. No information on environmental management, social responsibility or sustainability reporting. No involvement in sustainability programs, partnerships or industry initiatives.
    2. Minimal commitment to sustainability: Limited mention of sustainability practices, possibly only superficial. Basic information about environmental management without specific goals or measurable indicators. Low involvement in social initiatives or limited transparency about social impact.
    3. Average commitment to sustainability: Clear mention of sustainability initiatives, policies and objectives on the website. Moderate performance data on environmental management and social responsibility, possibly with some transparency. Some involvement in sustainability programs, although not leading within the sector.
    4. Significant commitment to sustainability: Detailed description of sustainability strategies, including measurable goals and performance indicators. Strong performance data on environmental management and social responsibility, with clear reporting on progress. Active involvement in sustainability initiatives and partnerships, with impact within the sector.
    5. Excellent commitment to sustainability: Progressive sustainability policies, including innovative approaches and best practices. Excellent performance data in environmental management, social responsibility and transparent reporting. Sustainability leadership, with active involvement in cross-sector initiatives and positive influence on the industry.

    Based on the text above, what is the sustainability score of the company? (Answer with a number from 1 to 5)

    Provide only a single digit as your answer.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
            n=5
        )
        scores = [re.findall(r'\d', choice['message']['content'].strip())[0] for choice in response['choices']]
        most_common_score = Counter(scores).most_common(1)[0][0]
        return {"sustainability_score_with_context": most_common_score}
    except Exception as e:
        logging.error(f"Error while analyzing sustainability: {e}")
        return {"sustainability_score_with_context": 1}

# OpenAI API call voor duurzaamheid (zonder context)
def analyze_sustainability_without_context(url):
    prompt = f"""
    Given the homepage URL '{url}', and based on general data available from sustainability indexes, reporting platforms, and news mentions, assess the company's commitment to sustainability. Consider the following points:
    - Participation in global sustainability initiatives or agreements.
    - Presence of sustainability or corporate responsibility reports in recognized databases.
    - Visibility in news related to positive sustainability practices and achievements.
    - Reported involvement in projects or partnerships for environmental or social benefits.
    
    Please rate the sustainability on a scale from 1 to 5, where:
    1. No importance to sustainability: No mention of sustainability initiatives, policies or objectives on the website. No information on environmental management, social responsibility or sustainability reporting. No involvement in sustainability programs, partnerships or industry initiatives.
    2. Minimal commitment to sustainability: Limited mention of sustainability practices, possibly only superficial. Basic information about environmental management without specific goals or measurable indicators. Low involvement in social initiatives or limited transparency about social impact.
    3. Average commitment to sustainability: Clear mention of sustainability initiatives, policies and objectives on the website. Moderate performance data on environmental management and social responsibility, possibly with some transparency. Some involvement in sustainability programs, although not leading within the sector.
    4. Significant commitment to sustainability: Detailed description of sustainability strategies, including measurable goals and performance indicators. Strong performance data on environmental management and social responsibility, with clear reporting on progress. Active involvement in sustainability initiatives and partnerships, with impact within the sector.
    5. Excellent commitment to sustainability: Progressive sustainability policies, including innovative approaches and best practices. Excellent performance data in environmental management, social responsibility and transparent reporting. Sustainability leadership, with active involvement in cross-sector initiatives and positive influence on the industry.

    Provide only a single digit as your answer.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
            n=5
        )
        scores = [int(choice['message']['content'].strip()) for choice in response['choices']]
        most_common_score = Counter(scores).most_common(1)[0][0]
        return {"sustainability_score_without_context": most_common_score}
    except Exception as e:
        logging.error(f"Error while analyzing sustainability without context: {e}")
        return {"sustainability_score_without_context": 1}  

# OpenAI API call voor industrie (zonder context)
def predict_industry_without_context(url):
    prompt = f"""
    Given the homepage URL '{url}', and based on general data online, predict which industry the company belongs to. The answer should be one of these industries only: Technology and Information Technology, Healthcare and Pharmaceuticals, Automotive, Energy, Retail.

    Simply answer with one of those options, nothing else. Also no sentences. 

    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
            n=1  
        )
        industry = response['choices'][0]['message']['content'].strip()
        return industry
    except Exception as e:
        logging.error(f"Error while predicting industry: {e}")
        return "Unknown"

# OpenAI API call voor industrie (met context)
def predict_industry(content):
    prompt = f"""
    Based on the following content, predict which industry the company belongs to. The answer should be one of these industries only: Technology and Information Technology, Healthcare and Pharmaceuticals, Automotive, Energy, Retail.

    Simply answer with one of those options, nothing else. Also no sentences. 

    Content: {content}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.0,
            n=1  
        )
        industry = response['choices'][0]['message']['content'].strip()
        return industry
    except Exception as e:
        logging.error(f"Error while predicting industry: {e}")
        return "Unknown"

# Clusteren van duurzaamheid-teksten
def cluster_texts(texts, num_clusters=8):
    logging.info("Loading sentence transformer model")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentence_embeddings = model.encode(texts)
    logging.info("Model loaded and sentences encoded")

    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(sentence_embeddings)
    cluster_labels = kmeans.labels_
    logging.info("Clustering completed")

    return cluster_labels

# Hoofdfunctie waarin alles samenkomt
def process_websites(file_path, output_path):

    # Logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Data inlezen
    df = pd.read_csv(file_path)
    results = []
    combined_texts = [] 

    for index, row in df.iterrows():
        url = row['url']
        logging.info(f"Processing URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                filtered_links = filter_links_by_keyword(soup, url)

                # scrapen van de gefilterde links
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

                combined_texts.append(combined_text)  

                # functies uitvoeren
                context_score = analyze_sustainability(combined_text)
                no_context_score = analyze_sustainability_without_context(url)
                industry_links = filter_links_for_industry(soup, url, max_links=2)
                industry_content = scrape_and_combine_text(industry_links)
                predicted_industry_with_context = predict_industry(industry_content)
                predicted_industry_without_context = predict_industry_without_context(url)

                # alles samenvoegen voor output
                result = {
                    'url': url,
                    'sustainability_relevant_urls': ';'.join(filtered_links),
                    'industry_relevant_urls': ';'.join(industry_links),
                    'scraped_text_for_sustainability': combined_text,
                    'scraped_text_for_industry': industry_content,
                    **context_score,
                    **no_context_score,
                    'predicted_industry_with_context': predicted_industry_with_context,
                    'predicted_industry_without_context': predicted_industry_without_context
                }
                results.append(result)
            else:
                logging.error(f"Failed to access {url}: HTTP {response.status_code}")
        except requests.Timeout:
            logging.error(f"Timeout occurred when accessing {url}")
        except requests.RequestException as e:
            logging.error(f"Network error processing {url}: {e}")

    # teksten clusteren
    cluster_labels = cluster_texts(combined_texts) 
    for result, label in zip(results, cluster_labels):
        result['cluster'] = label

    # wegschrijven naar csv bestand
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    process_websites("websites_to_scrape.csv", "final_results2.csv")