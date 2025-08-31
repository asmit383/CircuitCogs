import requests
import json
import sys
import os

# Configuration
url = 'http://localhost:8080/api/analyze'
image_path = r'/Users/asmitroy/Downloads/Fire_Analysis/1.png'

def analyze_fire_image(image_path):
    """Send an image to the fire analysis API and return the results
    
    Args:
        image_path (str): Path to the image file to analyze
        
    Returns:
        dict: The analysis results or error information
    """
    try:
        # Check if the image file exists
        if not os.path.exists(image_path):
            return {"error": f"Image file not found: {image_path}"}
            
        # Open the image file and prepare the request
        with open(image_path, 'rb') as img_file:
            files = {'file': img_file}
            
            # Send the request to the API
            print(f"Sending request to {url}...")
            response = requests.post(url, files=files, timeout=30)
            
            # Check if the request was successful
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API request failed with status code: {response.status_code}", "details": response.text}
                
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused. Make sure the server is running at the correct address and port."}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. The server might be busy or not responding."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def print_formatted_results(results):
    """Print the analysis results in a formatted way
    
    Args:
        results (dict): The analysis results from the API
    """
    if "error" in results:
        print(f"\nERROR: {results['error']}")
        if "details" in results:
            print(f"Details: {results['details']}")
        return
        
    if not results.get("fire_detected", False):
        print("\nNo fire detected in the image.")
        return
        
    print("\n===== FIRE ANALYSIS RESULTS =====")
    print(f"Top Material: {results.get('top_material', 'Unknown')}")
    print(f"Confidence: {results.get('confidence', 0):.2f}%")
    print(f"Recommended Extinguisher: {results.get('extinguisher', 'Unknown')}")
    
    print("\nAll Detected Materials:")
    for material in results.get('all_results', []):
        print(f"  - {material.get('material', 'Unknown')}: {material.get('similarity', 0):.2f}%")
    
    print("\nAI Reasoning:")
    print(f"  {results.get('ai_reasoning', 'No reasoning provided')}")
    
def main():
    # Allow specifying a different image via command line argument
    img_path = sys.argv[1] if len(sys.argv) > 1 else image_path
    
    # Analyze the image
    results = analyze_fire_image(img_path)
    
    # Print the results
    print_formatted_results(results)
    
    # Also print the raw JSON for reference
    print("\nRaw JSON Response:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()