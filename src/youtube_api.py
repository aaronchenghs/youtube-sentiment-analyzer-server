from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from constants import credentials_path, SCOPES, redirect_uri, service_account_file


def create_youtube_client():
    credentials = service_account.Credentials.from_service_account_file(
        service_account_file, scopes=SCOPES)

    youtube = build('youtube', 'v3', credentials=credentials)

    return youtube


def extract_video_id(url):
    # Extract the video ID from the YouTube URL
    import re
    regex = r"(?<=v=)[^&#]+"
    matches = re.search(regex, url)
    if matches:
        return matches.group(0)
    else:
        regex = r"(?<=be/)[^&#]+"
        matches = re.search(regex, url)
        return matches.group(0) if matches else None


def fetch_comments(youtube, video_id):
    comments = []
    next_page_token = None

    while len(comments) < 1000:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,  # Batch by 100 comments
            pageToken=next_page_token,
            textFormat="plainText"
        )
        response = request.execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            if len(comments) >= 1000:
                break

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return comments


def fetch_video_details(youtube, video_id):
    # Fetch video details including the category ID
    video_request = youtube.videos().list(
        part="snippet",
        id=video_id
    )
    video_response = video_request.execute()

    if 'items' in video_response and len(video_response['items']) > 0:
        video_title = video_response['items'][0]['snippet']['title']
        category_id = video_response['items'][0]['snippet']['categoryId']

        # Fetch the category name using the category ID
        category_request = youtube.videoCategories().list(
            part="snippet",
            id=category_id
        )
        category_response = category_request.execute()

        if 'items' in category_response and len(category_response['items']) > 0:
            video_genre = category_response['items'][0]['snippet']['title']
        else:
            video_genre = "Unknown Genre"

        return video_title, video_genre
    else:
        return "Unknown Title", "Unknown Genre"


