def analyze_comments(comments, trained_model):
    """
    Analyze the sentiment of a list of comments using a trained model.

    Parameters:
    - comments: list, a list of comments (strings) to be analyzed.
    - trained_model: a trained machine learning model for sentiment analysis.

    Returns:
    - analysis_results: dict, a dictionary containing the analysis summary.
    """

    positive_count = 0
    negative_count = 0

    for comment in comments:
        # Predict sentiment for each comment
        prediction = trained_model.predict([comment])
        if prediction == 0:
            positive_count += 1
        else:
            negative_count += 1

    # Calculate overall sentiment ratio or percentage
    total_comments = len(comments)
    positive_ratio = positive_count / total_comments if total_comments > 0 else 0
    negative_ratio = negative_count / total_comments if total_comments > 0 else 0

    # Prepare the analysis results
    analysis_results = {
        "total_comments": total_comments,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio
    }

    return analysis_results
