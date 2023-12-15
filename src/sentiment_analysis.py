def analyze_comments(comments, trained_model):
    """
    Analyze the sentiment of a list of comments using a trained multi-label model.

    Parameters:
    - comments: list, a list of comments (strings) to be analyzed.
    - trained_model: a trained machine learning model for sentiment analysis.

    Returns:
    - analysis_results: dict, a dictionary containing the analysis summary for each label.
    """
    label_names = ['IsToxic', 'IsAbusive', 'IsThreat', 'IsProvocative', 'IsObscene',
                   'IsHatespeech', 'IsRacist', 'IsNationalist', 'IsSexist',
                   'IsHomophobic', 'IsReligiousHate', 'IsRadicalism']

    label_counts = {label: {"positive": 0, "negative": 0} for label in label_names}
    label_comments = {label: [] for label in label_names}  # New dictionary for storing comments

    for comment in comments:
        predictions = trained_model.predict([comment])[0]

        for label, prediction in zip(label_names, predictions):
            if prediction == 0:  # 0 is positive, 1 is negative
                label_counts[label]["positive"] += 1
            else:
                label_counts[label]["negative"] += 1
                label_comments[label].append(comment)  # Append comment to respective label

    total_comments = len(comments)
    filtered_label_counts = {label: count for label, count in label_counts.items() if count["negative"] != 0}

    analysis_results = {
        "total_comments": total_comments,
        "details": filtered_label_counts,
        "label_comments": label_comments
    }
    return analysis_results
