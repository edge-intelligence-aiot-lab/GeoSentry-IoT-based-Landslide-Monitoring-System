from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "GeoSentry Dashboard is running!"

@app.route("/api/health")
def health():
    return jsonify({"status": "online"})

if __name__ == "__main__":
    app.run(debug=True)