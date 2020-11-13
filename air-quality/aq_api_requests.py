import requests

status = requests.get('https://aqs.epa.gov/data/api/metaData/isAvailable')
print(type(status))