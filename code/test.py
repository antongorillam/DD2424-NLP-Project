import requests

BASE = "http://127.0.0.1:8081/"

response = requests.get(BASE + "Synthesize")
print(response.json())
