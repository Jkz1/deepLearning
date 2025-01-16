from flask import Flask, jsonify, request
from flask_cors import CORS
from utils import execute

app = Flask(__name__)
CORS(app)

@app.route("/uploadImage", methods=['POST'])
def upload_image():
    file = request.files['file']
    if file:
        file.save("image.png")
        predict_result = execute()
        return jsonify({
            "message" : "Image predict success",
            "images" : predict_result['image'],
            "label" : predict_result['label'],
        })
    else:
        return jsonify({"message": "No image found in request"})

if __name__ == "__main__":
    app.run(debug=True, port=3050)
