import requests

# Set up the URL and payload
url = 'http://localhost:8888/parivategpt'
payload = {'query': 'Who controls Chernobyl?'}

# Send the POST request
response = requests.post(url, json=payload)

# Get the response data as JSON
data = response.json()

# Extract the output string
output_string = data['result']
sources = data['sources']

print("Response:"+output_string)  # Print the output string received from the API
print("\n\n\nSources:"+sources)
