import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch  # must be imported before utils/requests to avoid DLL conflicts on Windows
import os
import json
import time
import argparse
import yaml
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from dotenv import dotenv_values
from utils import MillionaireClient, AuthenticationError

ONLINE            = True
#RESULTS_FILE      = "results/Math/63Q-local-math-results.csv"
RESULTS_FILE      = "results/results.csv"
CONFIG_DIR        = "config"
ONLINE_TIMEOUT_S  = 30   # hard limit imposed by the quiz server
OFFLINE_TIMEOUT_S = 180  # generous limit for local runs to better analyze behaviour


def load_model(config_path: str):
    """Load a model from a YAML config file. Returns (model, model_key)."""
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_key        = os.path.splitext(os.path.basename(config_path))[0]
    model_class_name = config.pop("model_class", "LLMModel")

    if model_class_name == "RandomModel":
        from models.random import RandomModel
        return RandomModel(), model_key
    elif model_class_name == "MathLLMModel":
        from models.MATH import MathLLMModel
        return MathLLMModel(**config), model_key
    elif model_class_name == "MathDualModel":
        from models.MATHDUAL import MathDualModel
        phi_path       = os.path.join(CONFIG_DIR, config["phi_model"]       + ".yaml")
        mathstral_path = os.path.join(CONFIG_DIR, config["mathstral_model"] + ".yaml")
        phi_model, _       = load_model(phi_path)
        mathstral_model, _ = load_model(mathstral_path)
        return MathDualModel(phi_model, mathstral_model), model_key
    elif model_class_name == "SwitchModel":
        from models.SWITCH import SwitchModel
        general_path = os.path.join(CONFIG_DIR, config["general_model"] + ".yaml")
        math_path    = os.path.join(CONFIG_DIR, config["math_model"]    + ".yaml")
        math_comp_id = config.get("math_comp_id", 3)
        general_model, _ = load_model(general_path)
        math_model, _    = load_model(math_path)
        return SwitchModel(general_model, math_model, math_comp_id=math_comp_id), model_key
    else:
        from models.LLM import LLMModel
        return LLMModel(**config), model_key


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
    info  = model.get_info() if hasattr(model, "get_info") else {}
    label = "MATH BENCHMARK" if math_only else "MODEL"

    lines = ["=" * 80]
    lines.append(f"{label}: {model_key} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    for k, v in info.items():
        lines.append(f"{k}: {v}")
    lines.append("-" * 80)
    lines.append("Competition,Games,Avg Correct / 15,Accuracy %")

    total_games = total_correct = total_possible = 0

    for comp_id, data in play_history.items():
        n = data["num_games"]
        if n == 0:
            continue
        avg_correct = sum(data["level"]) / n
        max_lvl     = competitions[comp_id].max_levels
        accuracy    = avg_correct / max_lvl * 100
        total_games   += n
        total_correct += sum(data["level"])
        total_possible += max_lvl * n
        lines.append(f"{competitions[comp_id].name},{n},{avg_correct:.2f},{accuracy:.1f}%")

    if total_games > 0 and not math_only:
        overall_accuracy = total_correct / total_possible * 100
        lines.append(f"OVERALL,{total_games},{total_correct/total_games:.2f},{overall_accuracy:.1f}%")

    lines += ["=" * 80, ""]

    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
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

    competitions     = client.competitions.list_all()
    num_competitions = len(competitions)
    play_history     = {i: {"num_games": 0, "level": [], "score": []} for i in range(num_competitions)}
    seen_questions   = []   # collects every question seen this session

    print("\n=== Available Competitions ===")
    for comp in competitions:
        print(f"  {comp.id}: {comp.name} ({comp.max_levels} questions)")

    if math_only:
        math_comp_id = next(
            (i for i, c in enumerate(competitions) if "math" in c.name.lower()), None
        )
        if math_comp_id is None:
            print("Error: no math competition found.")
            return
        print(f"\nMath-only mode: running {multiplicity} games on '{competitions[math_comp_id].name}'")

    cnt        = -1
    print_cond = (not test_all and not math_only) or verbose

    while True:
        if math_only:
            cnt += 1
            comp_id = math_comp_id
            play_history[comp_id]["num_games"] += 1
            if verbose: print(f"Competition selected: {competitions[comp_id].name}")
        elif test_all:
            cnt += 1
            comp_id = cnt % num_competitions
            play_history[comp_id]["num_games"] += 1
            if verbose: print(f"Competition selected: {competitions[comp_id].name}")
        else:
            comp_id = int(input("Enter competition ID:"))
            while comp_id < 0 or comp_id >= num_competitions:
                print(f"Invalid competition. Please enter a competition ID in [0-{num_competitions-1}]")
                comp_id = int(input("Enter competition ID:"))
            play_history[comp_id]["num_games"] += 1

        if hasattr(model, "set_competition"):
            model.set_competition(comp_id)

        game = client.game.start(competition_id=comp_id)
        if print_cond:
            print(f"\n=== Starting Game ===\nSession ID: {game.session_id}")
            print(f"Total number of questions: {game.state.competition.max_levels}\n")

        while game.in_progress:
            question = game.current_question
            if not question:
                if print_cond: print("No question available. Game may have ended.")
                break

            if print_cond:
                print(f"\n--- Level {game.current_level} ---")
                print(f"Q: {question.text}\n")
                for opt in question.options:
                    print(f"  [{opt.id}] {opt.text}")

            options   = {str(opt.id): opt.text for opt in question.options}
            answer_id = model.answer(question.text, options)
            if print_cond: print(f"\nSelected answer: {answer_id}")

            time_left = game.time_remaining
            if time_left and print_cond:
                print(f"Time to answer: {30.0 - time_left:.1f}s")

            result = game.answer(answer_id)

            if hasattr(model, "log_result"):
                model.log_result(result.correct)

            if "math" in competitions[comp_id].name.lower():
                seen_questions.append({
                    "competition": competitions[comp_id].name,
                    "question":    question.text,
                    "options":     options,
                    "our_answer":  answer_id,
                    "correct":     result.correct,
                    "answer":      answer_id if result.correct else None,
                })

            if print_cond:
                if result.correct:
                    print("\n CORRECT!")
                    if result.game_over:
                        print(f"\n CONGRATULATIONS! You completed the game!")
                        print(f" Final earnings: ${result.earned_amount:,.2f}")
                    else:
                        print(f" Earned so far: ${result.earned_amount:,.2f}")
                elif result.timed_out:
                    print(f"\nTIMED OUT!\n Game Over!\n Final earnings: ${result.earned_amount:,.2f}")
                else:
                    print(f"\n WRONG ANSWER!\n Game Over!\n Final earnings: ${result.earned_amount:,.2f}")

        if print_cond:
            print(f"\n=== Game Summary ===\nReached Level: {game.current_level}")
            print(f"Total Earnings: ${game.earned_amount:,.2f}\n")

        play_history[comp_id]["level"].append(game.current_level - 1)
        play_history[comp_id]["score"].append(game.earned_amount)

        if math_only:
            if cnt + 1 >= multiplicity: break
        elif test_all:
            if cnt + 1 >= num_competitions * multiplicity: break
        else:
            if input("Play again? [Y/N]: ").lower() == 'n': break

    print("\n=== Session Summary ===")
    for comp_id, data in play_history.items():
        n = data["num_games"]
        if n == 0: continue
        print(f"\nCOMPETITION {competitions[comp_id].name}:")
        print(f"Number of games: {n}")
        print(f"Average correct answers: {sum(data['level'])/n:,.2f}")
        print(f"Average earnings: {sum(data['score'])/n:,.2f}")
        print()

    if output_csv:
        save_results(model_key, model, competitions, play_history, math_only)

    if seen_questions:
        os.makedirs("data/online_sessions", exist_ok=True)
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path    = f"data/online_sessions/session_{timestamp}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(seen_questions, f, ensure_ascii=False, indent=2)
        correct_count = sum(1 for q in seen_questions if q["correct"])
        print(f"\nSaved {len(seen_questions)} questions ({correct_count} with known correct answer) -> {out_path}")


def play_offline(model, model_key, test_all, multiplicity, verbose, output_csv, math_only, dataset):
    """
    Runs the model against a local JSON dataset of questions.
    Each question is answered multiplicity times and results are aggregated.
    """
    if not dataset:
        print("Error: --dataset <path> is required in offline mode.")
        return

    with open(dataset, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"\nOffline mode: {len(questions)} questions × {multiplicity} run(s) = {len(questions) * multiplicity} total answers\n")

    total = correct_count = 0

    for run in range(multiplicity):
        if multiplicity > 1:
            print(f"\n--- Run {run + 1} / {multiplicity} ---")
        for i, q in enumerate(questions):
            options = {str(k): v for k, v in q["options"].items()}

            if verbose:
                print(f"\nQ{i+1}: {q['question'][:80]}...")
                for k, v in options.items():
                    print(f"  [{k}] {v}")

            t0 = time.time()
            timed_out = False
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(model.answer, q["question"], options)
                try:
                    answer_id = future.result(timeout=OFFLINE_TIMEOUT_S)
                except FuturesTimeoutError:
                    answer_id = None
                    timed_out = True
            elapsed = time.time() - t0

            correct = (answer_id == q["answer"]) and not timed_out

            if hasattr(model, "log_result"):
                model.log_result(correct)

            total         += 1
            correct_count += int(correct)

            if timed_out:
                verdict = f"TIMEOUT after {elapsed:.1f}s"
            elif correct:
                verdict = f"CORRECT ({elapsed:.1f}s)"
            else:
                verdict = f"WRONG -> correct: {q['answer']} ({elapsed:.1f}s)"

            print(f"  Q{i+1:02d} -> answered {answer_id} | {verdict}")

    accuracy = correct_count / total * 100 if total else 0
    print(f"\n=== Offline Summary ===")
    print(f"Dataset : {dataset}")
    print(f"Model   : {model_key}")
    print(f"Score   : {correct_count}/{total} ({accuracy:.1f}%)")

    if hasattr(model, "finalize_log"):
        model.finalize_log(correct_count, total)

    if output_csv:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"OFFLINE | {model_key} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            info = model.get_info() if hasattr(model, "get_info") else {}
            for k, v in info.items():
                f.write(f"{k}: {v}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Dataset,Questions,Correct,Accuracy %\n")
            f.write(f"{dataset},{total},{correct_count},{accuracy:.1f}%\n")
            f.write("=" * 80 + "\n\n")
        print(f"Results appended to {RESULTS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="PoliMillionaire chatbot")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=f"Model config YAML filename (looked up in {CONFIG_DIR}/) or full path"
    )
    parser.add_argument("--test_all",    action="store_true", default=False, help="Test all competitions")
    parser.add_argument("--math",        action="store_true", default=False, help="Run on math competition only")
    parser.add_argument("--multiplicity",type=int, default=1, help="Games per competition")
    parser.add_argument("--verbose",     action="store_true", default=False, help="Print logs")
    parser.add_argument("--output_csv",  action="store_true", default=False,
                        help=f"Append session results to {RESULTS_FILE}")
    parser.add_argument("--dataset",     type=str, default=None,
                        help="Path to a JSON question dataset for offline mode")
    parser.add_argument("--debug",       action="store_true", default=False,
                        help="Log prompts, context, and answers to a timestamped file in logs/")
    args = parser.parse_args()

    if args.multiplicity < 1:
        print("Input Error: Multiplicity value should be greater or equal to 1!")
        return

    # Resolve config path: try as-is, then inside CONFIG_DIR/
    config_path = args.config
    if not os.path.exists(config_path):
        config_path = os.path.join(CONFIG_DIR, args.config)
        if not os.path.exists(config_path):
            # try adding .yaml extension
            config_path = config_path if config_path.endswith(".yaml") else config_path + ".yaml"
    if not os.path.exists(config_path):
        print(f"Error: config file '{args.config}' not found (looked in ./ and {CONFIG_DIR}/)")
        return

    model, model_key = load_model(config_path)
    if hasattr(model, "debug"):
        model.debug = args.debug

    if ONLINE:
        play_online(model, model_key, args.test_all, args.multiplicity, args.verbose, args.output_csv, args.math)
    else:
        play_offline(model, model_key, args.test_all, args.multiplicity, args.verbose, args.output_csv, args.math, args.dataset)


if __name__ == "__main__":
    main()
