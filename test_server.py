from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/test', methods=['POST'])
def test_endpoint():
    print("=" * 50)
    print("POST REQUEST RECEIVED:")
    print(f"Headers: {dict(request.headers)}")
    print(f"Content-Type: {request.content_type}")
    print(f"JSON Data: {request.get_json()}")
    print(f"Raw Data: {request.get_data()}")
    print("=" * 50)
    
    return jsonify({"status": "received", "message": "Data logged to console"}), 200

if __name__ == '__main__':
    print("Starting test server on http://localhost:8000")
    print("Endpoint: POST http://localhost:8000/test")
    app.run(host='localhost', port=8000, debug=True)
