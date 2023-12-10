from flask import Flask, request, jsonify

from constants import credentials_path
from src.model_training import get_or_train_model
from src.sentiment_analysis import analyze_comments
from src.youtube_api import create_youtube_client, extract_video_id, fetch_comments
import os

app = Flask(__name__)


@app.route('/analyze', methods=['POST'])
def analyze_video():
    data = request.json
    youtube_link = data['youtube_link']

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

    youtube = create_youtube_client()
    video_id = extract_video_id(youtube_link)

    # Fetch the top 1000 comments
    comments = fetch_comments(youtube, video_id)

    # Train the model or get the pre-trained model
    model = get_or_train_model()

    # Analyze the comments
    analysis_results = analyze_comments(comments, model)

    # Return the analysis results
    return jsonify(analysis_results)


if __name__ == '__main__':
    app.run(debug=True)