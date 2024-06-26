from src.model_training import  get_or_train_model
from src.sentiment_analysis import analyze_comments
from src.youtube_api import create_youtube_client, extract_video_id, fetch_comments
import os


def main():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\vojoh\\OneDrive\\Desktop\\Projects\\youtube-sentiment-analyzer-server\\csc3730-project-team-had-0548b315ad9b.json'

    youtube = create_youtube_client()

    video_link = input("Enter YouTube video link: ")
    video_id = extract_video_id(video_link)

    # Fetch the top 1000 comments
    comments = fetch_comments(youtube, video_id)

    # Train the model
    model = get_or_train_model()

    # Analyze the comments
    analysis_results = analyze_comments(comments, model)

    # Print the analysis results
    print(f"Most recent {analysis_results['total_comments']} comments analyzed.")
    print(f"{analysis_results['positive_count']} comments determined to have POSITIVE sentiment.")
    print(f"{analysis_results['negative_count']} comments determined to have NEGATIVE sentiment.")
    print(
        f"Overall Sentiment Rating: Positive - {analysis_results['positive_ratio'] * 100:.2f}%, Negative - {analysis_results['negative_ratio'] * 100:.2f}%")


if __name__ == "__main__":
    main()
