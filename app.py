from flask import Flask, request, jsonify

from constants import credentials_path
from src.model_training import get_or_train_model
from src.sentiment_analysis import analyze_comments
from src.youtube_api import create_youtube_client, extract_video_id, fetch_comments, fetch_video_details
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/analyze', methods=['POST'])
def analyze_video():
    data = request.json
    youtube_link = data['youtube_link']

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
    youtube = create_youtube_client()
    video_id = extract_video_id(youtube_link)

    video_title, video_genre = fetch_video_details(youtube, video_id)

    # Fetch the top 1000 comments
    comments = fetch_comments(youtube, video_id)

    # Train the model or get the pre-trained model
    model = get_or_train_model(video_genre)

    analysis_results = analyze_comments(comments, model)

    response_data = {
        "video_title": video_title,
        "analysis_results": {
            "total_comments": analysis_results["total_comments"],
            "details": analysis_results["details"],
            "label_comments": analysis_results["label_comments"]
        },
        "video_genre": video_genre
    }

    # Return the response
    return jsonify(response_data)


if __name__ == '__main__':
    app.run(debug=True)
