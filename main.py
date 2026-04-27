import torch  # must be imported before utils/requests to avoid DLL conflicts on Windows
import argparse
import importlib
from datetime import datetime
from dotenv import dotenv_values
from utils import MillionaireClient, AuthenticationError

ONLINE = True
MODELS = {
    "random":    ("models.random", "RandomModel"),
    "llama-1b":  ("models.LLM", "Llama1B"),
    "llama-3b":  ("models.LLM", "Llama3B"),
    "llama-8b":  ("models.LLM", "Llama8B"),
    "qwen-1.5b":       ("models.LLM", "Qwen15B"),
    "qwen-7b":         ("models.LLM", "Qwen7B"),
    "qwen-7b-code":    ("models.LLM", "QwenWithCode"),
}

RESULTS_FILE = "results/math-results.csv"


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
        return None
    return client


def save_results(model_key, model, competitions, play_history, math_only):
    info = model.get_info() if hasattr(model, "get_info") else {}

    lines = []
    lines.append("=" * 80)
    label = "MATH BENCHMARK" if math_only else "MODEL"
    lines.append(f"{label}: {model_key} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for k, v in info.items():
        lines.append(f"{k}: {v}")
    lines.append("-" * 80)
    lines.append("Competition,Games,Avg Correct / 15,Accuracy %")

    total_games = 0
    total_correct = 0
    total_possible = 0

    for comp_id, data in play_history.items():
        n = data["num_games"]
        if n == 0:
            continue
        avg_correct = sum(data["level"]) / n
        max_lvl = competitions[comp_id].max_levels
        accuracy = avg_correct / max_lvl * 100
        total_games += n
        total_correct += sum(data["level"])
        total_possible += max_lvl * n
        lines.append(
            f"{competitions[comp_id].name},{n},{avg_correct:.2f},{accuracy:.1f}%"
        )

    if total_games > 0 and not math_only:
        overall_accuracy = total_correct / total_possible * 100
        lines.append(
            f"OVERALL,{total_games},{total_correct/total_games:.2f},{overall_accuracy:.1f}%"
        )

    lines.append("=" * 80)
    lines.append("")

    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nResults appended to {RESULTS_FILE}")


def play_online(model, model_key, test_all, multiplicity, verbose, output_csv, math_only):
    """
    Runs an interactive quiz session against the online platform.
    Allows the user to select a competition, play multiple games, and
    tracks performance history (games played, levels reached, scores).
    At the end of the session, prints a summary with average statistics
    for each competition played.
    """

    client = login()
    if client is None:
        print("Game arrested.")
        return

    competitions = client.competitions.list_all()
    num_competitions = len(competitions)
    play_history = {
        i: {"num_games": 0, "level": [], "score": []}
        for i in range(num_competitions)
    }

    # List available competitions
    print("\n=== Available Competitions ===")
    for comp in competitions:
        print(f"  {comp.id}: {comp.name} ({comp.max_levels} questions)")

    # Resolve math competition index
    if math_only:
        math_comp_id = next(
            (i for i, c in enumerate(competitions) if "math" in c.name.lower()),
            None
        )
        if math_comp_id is None:
            print("Error: no math competition found.")
            return
        print(f"\nMath-only mode: running {multiplicity} games on '{competitions[math_comp_id].name}'")

    cnt = -1
    print_cond = (not test_all and not math_only) or verbose

    while True:
        # Choose a competition ID
        if math_only:
            cnt += 1
            comp_id = math_comp_id
            play_history[comp_id]["num_games"] += 1
            if verbose:
                print(f"Competition selected: {competitions[comp_id].name}")
        elif test_all:
            cnt += 1
            comp_id = cnt % num_competitions
            play_history[comp_id]["num_games"] += 1
            if verbose:
                print(f"Competition selected: {competitions[comp_id].name}")
        else:
            comp_id = int(input("Enter competition ID:"))
            while comp_id < 0 or comp_id >= num_competitions:
                print(f"Invalid competition. Please enter a competition ID in [0-{num_competitions-1}]")
                comp_id = int(input("Enter competition ID:"))
            play_history[comp_id]["num_games"] += 1

        # Start the game
        game = client.game.start(competition_id=comp_id)
        if print_cond:
            print("\n=== Starting Game ===")
            print(f"Session ID: {game.session_id}")
            print(f"Total number of questions: {game.state.competition.max_levels}")
            print()

        # Play the game
        while game.in_progress:
            question = game.current_question
            if not question:
                if print_cond: print("No question available. Game may have ended.")
                break

            if print_cond:
                print(f"\n--- Level {game.current_level} ---")
                print(f"Q: {question.text}")
                print()
                for opt in question.options:
                    print(f"  [{opt.id}] {opt.text}")

            options = {str(opt.id): opt.text for opt in question.options}

            answer_id = model.answer(question.text, options)
            if print_cond: print(f"\nSelected answer: {answer_id}")

            time_left = game.time_remaining
            if time_left and print_cond:
                print(f"Time to answer: {30.0 - time_left:.1f}s")

            result = game.answer(answer_id)

            if print_cond:
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

        if print_cond:
            print("\n=== Game Summary ===")
            print(f"Reached Level: {game.current_level}")
            print(f"Total Earnings: ${game.earned_amount:,.2f}")
            print()

        play_history[comp_id]["level"].append(game.current_level - 1)
        play_history[comp_id]["score"].append(game.earned_amount)

        if math_only:
            if cnt + 1 >= multiplicity:
                break
        elif test_all:
            if cnt + 1 >= num_competitions * multiplicity:
                break
        else:
            play_again = input("Play again? [Y/N]: ")
            if play_again.lower() == 'n':
                break

    print("\n=== Session Summary ===")
    for comp_id in play_history.keys():
        num_games = play_history[comp_id]["num_games"]
        if num_games == 0:
            continue
        print(f"\nCOMPETITION {competitions[comp_id]}:")
        print(f"Number of games: {num_games}")
        print(f"Average correct answers: {sum(play_history[comp_id]['level'])/num_games:,.2f}")
        print(f"Average earnings: {sum(play_history[comp_id]['score'])/num_games:,.2f}")
        print()

    if output_csv:
        save_results(model_key, model, competitions, play_history, math_only)


def play_offline(model, model_key, test_all, multiplicity, verbose, output_csv, math_only):
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
    parser.add_argument(
        "--test_all",
        action="store_true",
        default=False,
        help="Test all competitions"
    )
    parser.add_argument(
        "--math",
        action="store_true",
        default=False,
        help="Run benchmark on the math competition only"
    )
    parser.add_argument(
        "--multiplicity",
        type=int,
        required=False,
        default=1,
        help="Number of games per competition (or total games when --math is set)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print logs"
    )
    parser.add_argument(
        "--output_csv",
        action="store_true",
        default=False,
        help=f"Append session results to {RESULTS_FILE}"
    )
    args = parser.parse_args()

    if args.multiplicity < 1:
        print("Input Error: Multiplicity value should be greater or equal to 1!")
        return

    module_path, class_name = MODELS[args.model]
    module = importlib.import_module(module_path)
    model = getattr(module, class_name)()

    if ONLINE:
        play_online(model, args.model, args.test_all, args.multiplicity, args.verbose, args.output_csv, args.math)
    else:
        play_offline(model, args.model, args.test_all, args.multiplicity, args.verbose, args.output_csv, args.math)


if __name__ == "__main__":
    main()
