import pandas as pd
import spacy
import numpy as np
import openai
import logging
from collections import Counter
import re
import time

# Spacy inladen
nlp = spacy.load("en_core_web_md")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API key
openai.api_key = "OPENAI-KEY"

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

    retries = 3
    for attempt in range(retries):
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
            result = {f"sustainability_score_with_context_{i+1}": score for i, score in enumerate(scores)}
            result["sustainability_score_with_context"] = most_common_score
            return result
        except openai.error.RateLimitError as e:
            logging.error(f"Rate limit exceeded: {e}. Retrying after delay...")
            time.sleep(5)  
        except Exception as e:
            logging.error(f"Error while analyzing sustainability: {e}")
            return {f"sustainability_score_with_context_{i+1}": 1 for i in range(5)}.update({"sustainability_score_with_context": 1})
    return {f"sustainability_score_with_context_{i+1}": 1 for i in range(5)}.update({"sustainability_score_with_context": 1})

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

    retries = 3
    for attempt in range(retries):
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
        except openai.error.RateLimitError as e:
            logging.error(f"Rate limit exceeded: {e}. Retrying after delay...")
            time.sleep(5)  
        except Exception as e:
            logging.error(f"Error while analyzing sustainability without context: {e}")
            return {"sustainability_score_without_context": 1}
    return {"sustainability_score_without_context": 1}

def predict_industry_without_context(url):
    prompt = f"""
    Given the homepage URL '{url}', and based on general data online, predict which industry the company belongs to. The answer should be one of these industries only: Technology and Information Technology, Healthcare and Pharmaceuticals, Automotive, Energy, Retail.

    Simply answer with one of those options, nothing else. Also no sentences. 
    """

    retries = 3
    for attempt in range(retries):
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
        except openai.error.RateLimitError as e:
            logging.error(f"Rate limit exceeded: {e}. Retrying after delay...")
            time.sleep(5)  
        except Exception as e:
            logging.error(f"Error while predicting industry: {e}")
            return "Unknown"
    return "Unknown"

def predict_industry(content):
    prompt = f"""
    Based on the following content, predict which industry the company belongs to. The answer should be one of these industries only: Technology and Information Technology, Healthcare and Pharmaceuticals, Automotive, Energy, Retail.

    Simply answer with one of those options, nothing else. Also no sentences. 

    Content: {content}
    """

    retries = 3
    for attempt in range(retries):
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
        except openai.error.RateLimitError as e:
            logging.error(f"Rate limit exceeded: {e}. Retrying after delay...")
            time.sleep(5)  
        except Exception as e:
            logging.error(f"Error while predicting industry: {e}")
            return "Unknown"
    return "Unknown"

def analyze_websites(input_path, output_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    df = pd.read_csv(input_path)
    results = []

    for index, row in df.iterrows():
        url = row['url']
        scraped_text_for_sustainability = row['scraped_text_for_sustainability']
        scraped_text_for_industry = row['scraped_text_for_industry']

        logging.info(f"Analyzing URL: {url}")

        context_scores = analyze_sustainability(scraped_text_for_sustainability) or {}
        no_context_score = analyze_sustainability_without_context(url) or {}
        predicted_industry_with_context = predict_industry(scraped_text_for_industry) or "Unknown"
        predicted_industry_without_context = predict_industry_without_context(url) or "Unknown"

        result = {
            'url': url,
            'sustainability_relevant_urls': row['sustainability_relevant_urls'],
            'industry_relevant_urls': row['industry_relevant_urls'],
            'scraped_text_for_sustainability': scraped_text_for_sustainability,
            'scraped_text_for_industry': scraped_text_for_industry,
            'sustainability_score_with_context_1': context_scores.get('sustainability_score_with_context_1'),
            'sustainability_score_with_context_2': context_scores.get('sustainability_score_with_context_2'),
            'sustainability_score_with_context_3': context_scores.get('sustainability_score_with_context_3'),
            'sustainability_score_with_context_4': context_scores.get('sustainability_score_with_context_4'),
            'sustainability_score_with_context_5': context_scores.get('sustainability_score_with_context_5'),
            'sustainability_score_with_context': context_scores.get('sustainability_score_with_context'),
            'sustainability_score_without_context': no_context_score.get('sustainability_score_without_context'),
            'predicted_industry_with_context': predicted_industry_with_context,
            'predicted_industry_without_context': predicted_industry_without_context
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    analyze_websites("scraped_data.csv", "analyzed_data.csv")