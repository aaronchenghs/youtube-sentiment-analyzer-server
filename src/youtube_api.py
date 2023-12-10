from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from constants import credentials_path, SCOPES, redirect_uri


def create_youtube_client():
    # Step 1: The flow will redirect the user to Google's authorization server
    flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES, redirect_uri=redirect_uri)
    auth_url, _ = flow.authorization_url(access_type='offline')
    print('Please go to this URL and authorize access:', auth_url)

    # Step 2: The user will get an authorization code. This code is used to get the access token.
    auth_code = input('Enter the authorization code: ')
    flow.fetch_token(code=auth_code)

    # Step 3: Create the YouTube client
    credentials = flow.credentials
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
