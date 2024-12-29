import requests
import time

def test_endpoints():
    base_url = "http://localhost:8000"
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    try:
        response = requests.get(base_url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Test ping endpoint
    print("\nTesting ping endpoint...")
    try:
        response = requests.get(f"{base_url}/ping")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

    # Test chat endpoint
    print("\nTesting chat endpoint...")
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"message": "test message"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Waiting 2 seconds for server to start...")
    time.sleep(2)
    test_endpoints() 