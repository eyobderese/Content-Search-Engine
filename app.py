from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from service import Search

PDFProcessor = Search.PDFProcessor

processor = PDFProcessor()
app = Flask(__name__)
CORS(app)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print(file_path)

        # Create PDFProcessor object and process the file
        result = processor.process_pdf(file_path)

        return jsonify({"file_path": file_path, "result": "sucess"}), 200

# create an end point which accept url and call web_scraper method from PDFProcessor and return the result back to the user


@app.route('/scrape', methods=['POST'])
def scrape():
    data = request.json
    url = data.get("url")
    print(url)

    result = processor.process_web_content(url)

    return jsonify({"result": "Sucess"}), 200


# create and end point to search using get_similar_articles from PDFProcessor method take query and k from the user and ask the method and return it back for the user in list format
@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get("query")
    k = data.get("top-k")

    result = processor.get_similar_articles(query, int(k))

    return jsonify({"result": result}), 200


if __name__ == '__main__':
    app.run(debug=True)
