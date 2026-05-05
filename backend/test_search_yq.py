from yahooquery import search

def test_search(query):
    print(f"Searching for: {query}")
    results = search(query)
    if 'quotes' in results:
        for q in results['quotes']:
            print(f"Symbol: {q.get('symbol')}, Name: {q.get('shortname')}, Exchange: {q.get('exchDisp')}")
    else:
        print("No results")

test_search("Tata Motors")
test_search("Reliance")
test_search("Apple")
