import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer


def scrape_and_chunk_web_content(url):
    """Scrapes web content and chunks it based on headings or paragraphs, creating a title for each chunk."""
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Initialize containers for chunks and titles
    chunks = []
    titles = []

    # Process headings and paragraphs to create chunks and titles
    current_chunk = ""
    current_title = ""

    for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p']):
        text = tag.get_text(strip=True)

        if tag.name in ['h1', 'h2', 'h3', 'h4']:
            if current_chunk:
                chunks.append(current_chunk)
                titles.append(current_title or "Untitled Section")
            current_title = text  # Set title to current heading text
            current_chunk = ""    # Reset current chunk for a new section
        else:
            current_chunk += " " + text

    # Append the final chunk if it exists
    if current_chunk:
        chunks.append(current_chunk)
        titles.append(current_title or "Untitled Section")
    # create a list of titles and chunks
    titles_content = []
    for i in range(len(titles)):
        titles_content.append([titles[i], chunks[i]])

    return titles_content


chunks = scrape_and_chunk_web_content(
    "https://en.wikipedia.org/wiki/Python_(programming_language)")

print(chunks[0])
