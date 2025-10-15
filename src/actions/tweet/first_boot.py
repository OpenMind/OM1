import logging
import os
import warnings

import openai
from dotenv import load_dotenv

from runtime.single_mode.config import RuntimeConfig


def send_first_boot_tweet(config: RuntimeConfig):
    """Send the first boot tweet."""
    load_dotenv()
    logging.info("Attempting to send first boot tweet...")
    try:
        # Initialize Twitter client
        logging.info("Loading environment variables...")
        api_key = os.getenv("TWITTER_API_KEY")
        if not api_key:
            logging.error("Twitter API credentials not found in .env file")
            return

        logging.info("Initializing Twitter client...")
        # Suppress tweepy warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SyntaxWarning)
            import tweepy  # type: ignore

            client = tweepy.Client(
                consumer_key=os.getenv("TWITTER_API_KEY"),
                consumer_secret=os.getenv("TWITTER_API_SECRET"),
                access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
                access_token_secret=os.getenv("TWITTER_ACCESS_TOKEN_SECRET"),
            )

        # Get agent name from config
        agent_name = config.name
        logging.info(f"Agent name from config: {agent_name}")

        # Generate tweet using OpenAI
        prompt = f"""Write a tweet announcing that {agent_name} has just been activated. The tweet should:
1. Be friendly and enthusiastic
2. Mention the agent's name
3. Reference @openmind_agi and omOS
4. Include relevant emojis
5. Include #AI #omOS hashtags
6. Stay under 280 characters"""

        openai.api_key = os.getenv("OPENAI_API_KEY")
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that writes engaging tweets.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.7,
        )

        tweet_text = completion.choices[0].message.content.strip()  # type: ignore

        logging.info(f"Attempting to send tweet: {tweet_text}")

        response = client.create_tweet(text=tweet_text)
        tweet_id = response.data["id"]
        tweet_url = f"https://twitter.com/user/status/{tweet_id}"
        logging.info(f"First boot tweet sent! URL: {tweet_url}")

    except Exception as e:
        logging.error(f"Failed to send first boot tweet: {str(e)}")
        # Print full exception details
        logging.exception("Full error details:")
