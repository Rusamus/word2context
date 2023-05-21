import json
import requests

CONTEXT_NUMBER = 10
url = 'https://context.reverso.net/translation/'

proxies = {
    ""
}

headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
           "Content-Type": "application/json; charset=UTF-8",
          'Accept-Encoding': 'gzip, deflate, br',
          'Accept-Language': 'ru-RU,ru;q=0.8,en-US;q=0.5,en;q=0.3',
          'Connection': 'keep-alive',
          'DNT': '1',
          'Host': 'context.reverso.net',
          'Sec-Fetch-Dest': 'document',
          'Sec-Fetch-Mode': 'navigate',
          'Sec-Fetch-Site': 'cross-site',
          'TE': 'trailers',
          'Upgrade-Insecure-Requests': '1',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/113.0'
}

def get_context(word):
    data = {'source_text': word,
        'target_text': '',
        'source_lang': 'en',
        'target_lang': 'fr'}

    r1 = requests.get(url, timeout = 5, allow_redirects = True, proxies = proxies, headers = headers)
    r2 = requests.post("https://context.reverso.net/bst-query-service",
                                    headers=headers,
                                    data=json.dumps(data),
                                    timeout = 5, allow_redirects = True, proxies = proxies)
    
    input_string = r2.text
    json_data = json.loads(input_string)
    word_context = [data["s_text"].lower().replace("<em>", "").replace("</em>", "").replace(".", "") for data in json_data["list"]]
    return word_context[:CONTEXT_NUMBER]

