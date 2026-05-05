from yahooquery import search
results = search("Tata Motors")
for q in results.get('quotes', []):
    print(q)
