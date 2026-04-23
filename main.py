import argparse
import importlib
from dotenv import dotenv_values
from utils import MillionaireClient, AuthenticationError

ONLINE = True
MODELS = { 
    "baseline": ("models.baseline", "BaselineModel") # Example
}

def login():
    """
    Authenticates the user with the MillionaireClient using credentials
    stored in the .env file (API_URL, USERNAME, PASSWORD).
    """

    config = dotenv_values(".env")

    client = MillionaireClient(config["API_URL"])
    try:
        user = client.login(config["USERNAME"], config["PASSWORD"])
        print(f"\nWelcome, {user.username}! (Role: {user.role})")
    except AuthenticationError as e:
        print(f"Login failed: {e}")

    return client

def play_online(model):
    """
    Runs an interactive quiz session against the online platform.
    Allows the user to select a competition, play multiple games, and
    tracks performance history (games played, levels reached, scores).
    At the end of the session, prints a summary with average statistics
    for each competition played.
    """

    client = login()

    # Session summary
    num_competitions = len(client.competitions.list_all())
    play_history = {
        str(i): {"num_games": 0, "level": [], "score": []} 
        for i in range(num_competitions)
    }

    while(True):
        # List available competitions
        print("\n=== Available Competitions ===")
        competitions = client.competitions.list_all()
        for comp in competitions:
            print(f"  {comp.id}: {comp.name} ({comp.max_levels} questions)")
        
        # Choose a competition ID
        comp_id = input("Enter competition ID:")
        play_history[comp_id]["num_games"] += 1

        # Start the game
        print("\n=== Starting Game ===")
        game = client.game.start(competition_id=comp_id)
        print(f"Session ID: {game.session_id}")
        print(f"Total number of questions: {game.state.competition.max_levels}")
        print()

        # Play the game
        while game.in_progress:
            question = game.current_question
            if not question:
                print("No question available. Game may have ended.")
                break

            print(f"\n--- Level {game.current_level} ---")
            print(f"Q: {question.text}")
            print()

            for opt in question.options:
                print(f"  [{opt.id}] {opt.text}")
            options = {
                str(opt.id) : opt.text
                for opt in question.options
            }

            # Get answer
            answer_id = model.answer(question.text, options)
            print(f"\nSelected answer: {answer_id}")

            # Get time remaining
            time_left = game.time_remaining
            if time_left:
                print(f"Time to answer: {30.0 - time_left:.1f}s")

            # Submit answer
            result = game.answer(answer_id)

            if result.correct:
                print("\n CORRECT!")
                if result.game_over:
                    print(f"\n CONGRATULATIONS! You completed the game!")
                    print(f" Final earnings: ${result.earned_amount:,.2f}")
                else:
                    print(f" Earned so far: ${result.earned_amount:,.2f}")
            elif result.timed_out:
                print("\nTIMED OUT!")
                print(f"\n Game Over!")
                print(f" Final earnings: ${result.earned_amount:,.2f}")
            elif not result.correct:
                print("\n WRONG ANSWER!")
                print(f"\n Game Over!")
                print(f" Final earnings: ${result.earned_amount:,.2f}")

        print("\n=== Game Summary ===")
        print(f"Reached Level: {game.current_level}")
        print(f"Total Earnings: ${game.earned_amount:,.2f}")
        print()

        play_history[comp_id]["level"].append(game.current_level - 1)
        play_history[comp_id]["score"].append(game.earned_amount)

        play_again = input("Play again? [Y/N]: ")
        if play_again.lower() == 'n':
            break
    
    print("\n=== Session Summary ===")
    for comp_id in play_history.keys():
        num_games = play_history[comp_id]["num_games"]
        if num_games == 0: continue

        print(f"\nCOMPETITION {comp_id}:")
        print(f"Number of games: {num_games}")
        print(f"Average correct answers: {sum(play_history[comp_id]["level"])/num_games:,.2f}")
        print(f"Average score: {sum(play_history[comp_id]["score"])/num_games:,.2f}")

def play_offline(model):
    """
    Runs the quiz session in offline mode using a local dataset.
    Used when the online platform is no longer available.
    """

    #TODO
    return

def main():
    parser = argparse.ArgumentParser(description="PoliMillionaire chatbot")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=MODELS.keys(),
        help=f"Model to use. Available: {list(MODELS.keys())}"
    )
    args = parser.parse_args()

    module_path, class_name = MODELS[args.model]
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    model = model_class()

    play_online(model) if ONLINE else play_offline(model)

if __name__ == "__main__":
    main()