from flask import Flask, render_template, request, jsonify
import os
import re
from datetime import datetime
import logging
import requests
import fitz
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from transformers import pipeline
import nltk
nltk.download('punkt_tab')
from nltk.sentiment import SentimentIntensityAnalyzer
from apscheduler.schedulers.background import BackgroundScheduler
from bs4 import BeautifulSoup
from apscheduler.schedulers.blocking import BlockingScheduler
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC

nltk.download('vader_lexicon')
nltk.download('punkt')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sia = SentimentIntensityAnalyzer()

SAVE_DIR = "reports"
SUMMARY_FOLDER = os.path.join(SAVE_DIR, 'summarized_reports')
os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)

def download_pdf(url, subfolder, filename):
    folder = os.path.join(SAVE_DIR, subfolder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    filepath = os.path.join(folder, filename)
    if not os.path.isfile(filepath):  # Check if file already exists
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {filepath}")
        else:
            print(f"Failed to download: {url} (Status code: {response.status_code})")
    else:
        print(f"File already exists: {filepath}")
    return filepath

def extract_text_from_pdf_safe(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text("text")
        
        if not text.strip():  # Check if text extraction is empty
            print(f"No text found in PDF: {pdf_path}. It may be an image-based PDF.")
        return text
    except Exception as e:
        print(f"Failed to extract text from PDF {pdf_path}: {e}")
        return None

def save_summary_report_in_folder(folder_name, filename, summary, sentiment):
    subfolder = os.path.join(SUMMARY_FOLDER, folder_name)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    
    summary_filepath = os.path.join(subfolder, filename)
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        f.write(f"SUMMARY OF {filename}:\n\n")
        f.write(f"Summary:\n{summary}\n\n")
        f.write(f"Sentiment Analysis:\n{sentiment}\n")
    
    print(f"Summary report saved: {summary_filepath}")
    
def generate_report_for_pdf(pdf_path, folder_name):
    text = extract_text_from_pdf_safe(pdf_path)
    if not text.strip():
        print(f"Warning: No text extracted from {pdf_path}. Skipping summarization.")
        return  # Skip processing if no text was extracted

    summary = summarize_large_text(text)
    if not summary.strip():
        print(f"Warning: No summary generated for {pdf_path}.")
        return  # Skip saving if no summary was generated

    sentiment = analyze_sentiment(text)
    save_summary_report_in_folder(folder_name, os.path.basename(pdf_path) + ".txt", summary, sentiment)

def process_existing_pdf(pdf_path, folder_name):
    summary_file_path = os.path.join(SUMMARY_FOLDER, folder_name, os.path.basename(pdf_path) + ".txt")
    
    # Check if the summary exists and if it's non-empty
    if os.path.isfile(summary_file_path):
        with open(summary_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if content:
            print(f"Summary already exists for: {pdf_path}")
            return  # Skip reprocessing if the summary is valid
    
    print(f"Generating summary for existing PDF: {pdf_path}")
    generate_report_for_pdf(pdf_path, folder_name)

def merge_summaries(folder_name, merged_filename):
    subfolder = os.path.join(SUMMARY_FOLDER, folder_name)
    merged_filepath = os.path.join(SUMMARY_FOLDER, merged_filename)
    
    if not os.path.exists(subfolder):
        print(f"No summaries found in folder: {subfolder}")
        return

    try:
        with open(merged_filepath, 'w', encoding='utf-8') as merged_file:
            for file in sorted(os.listdir(subfolder)):  # Sorting ensures consistent order
                if file.endswith(".txt"):
                    file_path = os.path.join(subfolder, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        merged_file.write(f"=== SUMMARY FROM {file} ===\n\n")
                        merged_file.write(f.read() + "\n\n")
        
        print(f"Merged summary saved: {merged_filepath}")
    except Exception as e:
        print(f"Error while merging summaries: {e}")

def summarize_large_text(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    chunks = [" ".join(sentences[i:i + 20]) for i in range(0, len(sentences), 20)]
    summaries = []
    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        except Exception as e:
            logging.error(f"Error summarizing chunk: {e}")
    return " ".join(summaries)

def analyze_sentiment(text):
    return sia.polarity_scores(text)

def save_summary_report(folder_name, filename, summary, sentiment):
    subfolder = os.path.join(SUMMARY_FOLDER, folder_name)
    os.makedirs(subfolder, exist_ok=True)
    summary_filepath = os.path.join(subfolder, filename)
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        f.write(f"SUMMARY:\n{summary}\n\n")
        f.write(f"SENTIMENT:\n{sentiment}\n")

def generic_scraper(folder_name, base_url):
    logging.info(f"Starting scraper for: {base_url}...")
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links = [a['href'] for a in soup.find_all("a", href=True) if a['href'].endswith('.pdf')]
        for link in pdf_links:
            full_url = link if link.startswith("http") else f"{base_url}/{link}"
            filename = os.path.basename(full_url)
            pdf_path = download_pdf(full_url, folder_name, filename)
            if pdf_path:
                text = extract_text_from_pdf_safe(pdf_path)
                if text:
                    summary = summarize_large_text(text)
                    sentiment = analyze_sentiment(summary)
                    save_summary_report(folder_name, filename + '.txt', summary, sentiment)
    except Exception as e:
        logging.error(f"Error scraping {base_url}: {e}")

# Fidelity scraper
def scrape_fidelity():
    folder_name = "fidelity_reports"
    print("Starting Fidelity scraper...")
    driver_path = r"C:\edgedriver_win64\msedgedriver.exe"
    service = Service(driver_path)
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    driver = webdriver.Edge(service=service, options=options)
    
    try:
        fidelity_url = 'https://fundresearch.fidelity.com/mutual-funds/analysis/316345305?documentType=QFR'
        driver.get(fidelity_url)
        
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        pdf_link = None
        for link in soup.find_all("a", href=True):
            if re.search(r'View\s*as\s*PDF', link.text, re.IGNORECASE):
                pdf_link = link['href']
                break

        if pdf_link:
            full_url = pdf_link if pdf_link.startswith('http') else f"https://fundresearch.fidelity.com{pdf_link}"
            filename = full_url.split('/')[-1]
            pdf_path = download_pdf(full_url, folder_name, filename)
            process_existing_pdf(pdf_path, folder_name)
            merge_summaries(folder_name, f"{folder_name}_summary.txt")
        else:
            print("No PDF download link found for Fidelity.")
    finally:
        driver.quit()

# Baron Capital scraper
def scrape_baron():
    folder_name = "baron_reports"
    print("Starting Baron Capital scraper...")
    driver_path = r"C:\edgedriver_win64\msedgedriver.exe"
    service = Service(driver_path)
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    driver = webdriver.Edge(service=service, options=options)
    
    try:
        baron_url = 'https://www.baroncapitalgroup.com/insights-webcasts?menu=Institutions#Reports'
        driver.get(baron_url)
        
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        report_links = [a['href'] for a in soup.find_all("a", href=True) if '.pdf' in a['href']]
        
        for link in report_links:
            full_url = link if link.startswith('http') else f"https://www.baroncapitalgroup.com{link}"
            filename = full_url.split('/')[-1]
            pdf_path = download_pdf(full_url, folder_name, filename)
            process_existing_pdf(pdf_path, folder_name)
        
        merge_summaries(folder_name, f"{folder_name}_summary.txt")
        print(f"Baron Capital scraping and summarization completed.")
    except Exception as e:
        print(f"Error while scraping Baron Capital: {e}")
    finally:
        driver.quit()

# Goldman Sachs scraper
def scrape_goldman():
    folder_name = "goldman_reports"
    print("Starting Goldman Sachs scraper...")
    driver_path = r"C:\edgedriver_win64\msedgedriver.exe"
    service = Service(driver_path)
    options = webdriver.EdgeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--log-level=3")
    driver = webdriver.Edge(service=service, options=options)
    
    try:
        goldman_url = 'https://www.goldmansachs.com/investor-relations/financials/'
        driver.get(goldman_url)
        
        WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find all links to PDF documents
        report_links = [a['href'] for a in soup.find_all("a", href=True) if '.pdf' in a['href']]
        
        for link in report_links:
            full_url = link if link.startswith('http') else f"https://www.goldmansachs.com{link}"
            filename = full_url.split('/')[-1]
            pdf_path = download_pdf(full_url, folder_name, filename)
            process_existing_pdf(pdf_path, folder_name)
        
        # Merge summaries
        merge_summaries(folder_name, f"{folder_name}_summary.txt")
        print(f"Goldman Sachs scraping and summarization completed.")
    
    except Exception as e:
        print(f"Error while scraping Goldman Sachs: {e}")
    finally:
        driver.quit()

# Scheduler setup
def schedule_scraping():
    """
    Schedule the scraping tasks for Fidelity, Baron Capital, and Goldman Sachs.
    Each task runs every 120 minutes and ensures that summaries are generated.
    """
    scheduler = BlockingScheduler()
    
    # Schedule Fidelity scraper
    scheduler.add_job(
        scrape_fidelity, 
        'interval', 
        minutes=120, 
        next_run_time=datetime.now(), 
        id="fidelity_scraper"
    )
    
    # Schedule Baron Capital scraper
    scheduler.add_job(
        scrape_baron, 
        'interval', 
        minutes=120, 
        next_run_time=datetime.now(), 
        id="baron_scraper"
    )
    
    # Schedule Goldman Sachs scraper
    scheduler.add_job(
        scrape_goldman, 
        'interval', 
        minutes=120, 
        next_run_time=datetime.now(), 
        id="goldman_scraper"
    )
    
    print("Scheduler started. Scraping tasks will run every 120 minutes.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")

# Individual scrapers for each company
def scrape_fidelity_reports():
    scrape_fidelity()

def scrape_baron_capital_reports():
    scrape_baron()

def scrape_goldman_sachs_reports():
    scrape_goldman()

def scrape_first_community_reports():
    generic_scraper("first_community_reports", "https://firstcommunitysc.q4ir.com/news-market-information/analyst-coverage/default.aspx")

def scrape_chesapeake_reports():
    generic_scraper("chesapeake_reports", "https://chesapeakefinancialshares.com/analyst-reports/default.aspx")

def scrape_oaktree_reports():
    generic_scraper("oaktree_reports", "https://www.oaktreecapital.com/insights")

def scrape_barclays_reports():
    generic_scraper("barclays_reports", "https://live.barcap.com/BC/barcaplive?menuCode=AR_EQ_PUB_ST")

def scrape_evercore_reports():
    generic_scraper("evercore_reports", "https://evercoreisi.mediasterling.com/fundamental/sector/158")

def scrape_morningstar_reports():
    generic_scraper("morningstar_reports", "https://my.pitchbook.com/research-center/1501933")

def scrape_hoisington_reports():
    generic_scraper("hoisington_reports", "https://hoisington.com/economic_overview.html")

def scrape_robotti_reports():
    generic_scraper("robotti_reports", "https://advisors.robotti.com/separately-managed-accounts/")

def scrape_behind_numbers_reports():
    generic_scraper("behind_numbers_reports", "https://btnresearch.com/btn-archive")

def scrape_jpmorgan_reports():
    generic_scraper("jpmorgan_reports", "https://am.jpmorgan.com/us/en/asset-management/adv/insights/market-insights/guide-to-the-markets")

# Map company names to their scrapers
company_scrapers = {
    "Fidelity": scrape_fidelity,
    "Baron Capital": scrape_baron,
    "Goldman Sachs": scrape_goldman,
    "First Community": scrape_first_community_reports,
    "Chesapeake Financial": scrape_chesapeake_reports,
    "Oaktree Capital": scrape_oaktree_reports,
    "Barclays Equity Strategy": scrape_barclays_reports,
    "Evercore ISI": scrape_evercore_reports,
    "Morningstar Research": scrape_morningstar_reports,
    "Hoisington": scrape_hoisington_reports,
    "Robotti Advisors": scrape_robotti_reports,
    "Behind the Numbers": scrape_behind_numbers_reports,
    "JPMorgan Guide to the Markets": scrape_jpmorgan_reports
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download():
    selected_companies = request.form.getlist("companies")
    results = []
    for company in selected_companies:
        if company in company_scrapers:
            logging.info(f"Executing scraper for: {company}")
            company_scrapers[company]()
            results.append(f"Scraper executed for: {company}")
    return jsonify({"results": results})

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    for company, job_logic in company_scrapers.items():
        scheduler.add_job(job_logic, 'interval', minutes=120, id=f"{company.lower().replace(' ', '_')}_scraper")
    scheduler.start()
    try:
        app.run(debug=True)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
