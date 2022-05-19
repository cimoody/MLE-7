
import secrets
import random

from typing import Dict, List

from praw import Reddit
from praw.models.reddit.subreddit import Subreddit
from praw.models import MoreComments

from transformers import pipeline


def get_subreddit(display_name:str) -> Subreddit:
    """Get subreddit object from display name

    Args:
        display_name (str): 
            This is the name in the reddit.com/r/display_name

    Returns:
        Subreddit: 
            Returns a subreddit with the specified display name
    """
    reddit = Reddit(
        client_id=secrets.REDDIT_API_CLIENT_ID,        
        client_secret=secrets.REDDIT_API_CLIENT_SECRET,
        user_agent=secrets.REDDIT_API_USER_AGENT
        )
    
    subreddit = reddit.subreddit(display_name) # YOUR CODE HERE
    return subreddit

def get_comments(subreddit:Subreddit, limit:int=3) -> List[str]:
    """ Get comments from subreddit

    Args:
        subreddit (Subreddit): 
            This takes the output from the get_subreddit function
        limit (int, optional): 
            This is number of comments to get from the subreddit. Defaults to 3.

    Returns:
        List[str]: List of comments
    """
    top_comments = []
    for submission in subreddit.top(limit=limit):
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            top_comments.append(top_level_comment.body)
    return top_comments

def run_sentiment_analysis(comment:str) -> Dict:
    """Run sentiment analysis on comment using default distilbert model
    
    Args:
        comment (str): This is the comment that the sentiment analysis classifies.
        
    Returns:
        str: Sentiment analysis result
    """
    sentiment_model = pipeline("sentiment-analysis")
    sentiment = sentiment_model(comment)
    return sentiment[0]


if __name__ == '__main__':
    submission = get_subreddit("TSLA") # Getting TSLA comments
    comments = get_comments(submission)
    comment = comments
    sentiment = run_sentiment_analysis(comment)
    
    print(f'The comment: {comment}')
    print(f'Predicted Label is {sentiment["label"]} and the score is {sentiment["score"]:.3f}')
