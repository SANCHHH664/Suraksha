import json
import urllib.request

url = 'http://127.0.0.1:5000/scan'
payload = json.dumps({'target': '127.0.0.1'}).encode('utf-8')
req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=10) as resp:
    print(resp.read().decode('utf-8'))
