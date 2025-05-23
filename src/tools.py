from unstructured.partition.html import partition_html
import requests
import json


tools_config = [
    {
        "type": "function",
        "function": {
            "name": "internet_search",
            "description": "Search the internet for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "The search query to use.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "The number of results to return.",
                    },
                },
                "required": ["search_query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_website",
            "description": "Scrape a website for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the website to scrape.",
                    },
                },
                "required": ["url"],
            },
        }
    }
]


def search_internet(ctx, query, num_results=5):
    serper_api_key = ctx.get_dependency("serper_api_key")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        "X-API-KEY": serper_api_key,
        "content-type": "application/json",
    }

    response = requests.post(url, headers=headers, data=payload)
    data = response.json()
    if "organic" not in data:
        return "No results, perhaps an API key is missing or the query is invalid."

    results = data["organic"]
    string_results = []
    for result in results[:num_results]:
        string_results.append(f"""Title: {result['title']}
        Link: {result['link']}
        Snippet: {result['snippet']}
        -----------------""")
    r = "\n".join(string_results)
    return r


def scrape_website(ctx, url):
    print("LOG: scraping website content")
    browserless_api_key = ctx.get_dependency("browserless_api_key")
    site_url = url
    api_url = f"https://chrome.browserless.io/content?token={browserless_api_key}"

    payload = json.dumps({"url": site_url})
    headers = {
        "cache-control": "no-cache", 
        "content-type": "application/json"
    }

    response = requests.post(api_url, headers=headers, data=payload)

    if response.status_code != 200:
        return f"error scraping website content: {response.status_code} {response.text}"
    if not response.text.strip():
        return "error scraping website content: empty response from website scraper"

    try:
        elements = partition_html(text=response.text)
    except Exception as e:
        return f"error parsing HTML: {str(e)}"

    content = "\n\n".join([str(el) for el in elements])
    content_chunks = [content[i : i + 8000] for i in range(0, len(content), 8000)]

    return "\n\n".join(content_chunks)
