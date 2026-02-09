"""
Simple test for NIM embeddings endpoint.

The gate route requires:
1. GET request to obtain JWT-APP cookie via redirect
2. POST request using the redirected URL and JWT-APP cookie
"""

import requests
import dtlpy as dl

dl.setenv('prod')


def call_nim_endpoint(app_id: str, endpoint: str, method: str = "get", data: dict = None):
    """
    Call a NIM endpoint through Dataloop's app routing.
    
    Args:
        app_id: Dataloop app ID
        endpoint: API endpoint (e.g., "/v1/embeddings", "/v1/health/live")
        method: HTTP method ("get" or "post")
        data: JSON data for POST requests
    
    Returns:
        Response object
    """
    app = dl.apps.get(app_id=app_id)
    
    # Get the route URL (without panel name suffix)
    route = list(app.routes.values())[0].rstrip('/')
    base_url = '/'.join(route.split('/')[:-1])
    
    if method.lower() == "get":
        # GET requests work directly with auth header
        url = f"{base_url}{endpoint}"
        return requests.get(url, headers=dl.client_api.auth)
    else:
        # POST requests need JWT-APP cookie from redirect
        session = requests.Session()
        
        # First GET to obtain JWT-APP cookie and follow redirect
        resp = session.get(base_url, headers=dl.client_api.auth)
        
        # Use the redirected URL and session cookies
        url = f"{resp.url.rstrip('/')}{endpoint}"
        return session.post(url, json=data, cookies=dict(session.cookies))


# Test it
if __name__ == "__main__":
    app_id = "69885809eb577aee86877518"
    model = "nvidia/nvclip-vit-h-14"
    
    # Test GET endpoint
    print("Testing GET /v1/manifest...")
    resp = call_nim_endpoint(app_id, "/v1/manifest", method="get")
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        print(f"Response: {resp.json()}")
    
    print()
    
    # Test POST endpoint
    print("Testing POST /v1/embeddings...")
    data = {
        "input": ["The quick brown fox jumped over the lazy dog"],
        "model": model,
        "encoding_format": "float"
    }
    resp = call_nim_endpoint(app_id, "/v1/embeddings", method="post", data=data)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        result = resp.json()
        print(f"Model: {result.get('model')}")
        print(f"Embeddings: {len(result.get('data', []))} items")
        for i, emb in enumerate(result.get('data', [])):
            print(f"  [{i}] {len(emb.get('embedding', []))} dimensions")
    else:
        print(f"Error: {resp.text[:200]}")
