import sys
import boto3
from dotenv import load_dotenv
from eval.game import (
    Game,
    Player1,
    Player2,
    random_bedrock_model,
)
from loguru import logger
import random

logger.remove()
logger.add(sys.stdout, level="INFO")

load_dotenv()


def main():
    # Environment Settings

    models = [
        "mistral_8x7b",
        "mistral_7b",
        "ai21_ultra",
        "ai21_mid",
        "claude_3_sonnet",
        "claude_3_haiku",
        "claude_2",
        # "claude_2_1", # Doesn't want to play
        "claude_instant",
        "cohere_command",
        "cohere_light",
        "titan_express",
        "titan_lite",
        # "llama2_13b", # not working
        # "llama2_70b", # not working
    ]

    random.seed()
    # Get a random model from the available models
    rand_model_1 = random.choice(models)

    # Remove the selected model from the available models list
    models.remove(rand_model_1)

    # Get another random model different from rand_model_1
    rand_model_2 = random.choice(models)

    # # force a match
    # rand_model_1 = "ai21_ultra"
    # rand_model_2 = "titan_express"

    game = Game(
        render=True,
        player_1=Player1(
            nickname="Bedrock",
            model=rand_model_1,
        ),
        player_2=Player2(
            nickname="PartyRock",
            model=rand_model_2,
        ),
    )
    return game.run()


if __name__ == "__main__":
    main()
