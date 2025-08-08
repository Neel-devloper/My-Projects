import requests
from bs4 import BeautifulSoup

API_key = 'AIzaSyBvFDwTSpxT-l4Q0F7L-2U-ObzrP3xk-dU'
SEARCH_ENGINE_ID = '1780952622f2444c3'

print("add '/images' to search for images at the end of your search")
print("add '/links' to search for links at the end of your search")
print("add '/videos' to search for videos at the end of your search\n")

search_query = input('Search Anything: ')
url = 'https://www.googleapis.com/customsearch/v1'

if '/images' in search_query.strip():
    params = {
        'q': search_query,
        'key': API_key,
        'cx': SEARCH_ENGINE_ID,
        'searchType': 'image'
    }
    response = requests.get(url, params=params)
    results = response.json()['items']
    for item in results:
        print(item['link'])
        print()

elif '/videos' in search_query.strip():
    search_query = search_query + ' youtube'
    params = {
        'q': search_query,
        'key': API_key,
        'cx': SEARCH_ENGINE_ID,
    }

    response = requests.get(url, params=params)
    results = response.json()['items']
    for item in results:
        print(item['link'])
        print()

elif '/links' in search_query.strip():
    params = {
        'q': search_query,
        'key': API_key,
        'cx': SEARCH_ENGINE_ID
    }

    response = requests.get(url, params=params)
    results = response.json().get('items', [])
    for item in results:
        print(item.get('link', 'No link found'))
        print()


else:
    search_query = search_query + ' wikipedia'
    params = {
        'q': search_query,
        'key': API_key,
        'cx': SEARCH_ENGINE_ID
    }

    response = requests.get(url, params=params)
    results = response.json()


    if 'items' in results:
        # scrape the website
        link = results['items'][0]['link'] # gets wikipedia link
        response = requests.get(link)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('h1').text
        print(f'Title: {title}')
        paragraphs = soup.find_all('p')
        for i, paragraph in enumerate(paragraphs):
            print(f'Paragraph {i+1}: {paragraph.text}')
    else:
        print('Error: No results found.')
