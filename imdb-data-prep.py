import requests
import tarfile


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
filename = "aclImdb_v1.tar.gz"

response = requests.get(url, stream=True)
response.raise_for_status()  # Raise an exception for bad status codes

with open(filename, "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print(f"Downloaded {filename}")

# filename = 'aclImdb_v1.tar.gz'
with tarfile.open(filename, 'r:gz') as tar:
    tar.extractall()

print("Extraction complete.")