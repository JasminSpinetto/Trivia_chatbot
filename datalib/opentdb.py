import argparse
import json
import requests
import time
import os

BASE_API_URL = "https://opentdb.com/api.php?amount=50&type=multiple"
TOKEN_URL = "https://opentdb.com/api_token.php?command=request"
time_limit = 5

def retrieve_trivia_db(save_path):
    """
    Retrieve and save an Open Trivia Database question set.

    This function requests a session token from the Open Trivia DB API,
    then uses that token to fetch multiple-choice questions in batches.
    It continues retrieving questions until the API indicates that all
    available questions for the requested query have been returned.

    The collected questions are saved as a JSON file named
    "trivia_db.json" under the provided save_path.
    """

    questions = []
    os.makedirs(save_path, exist_ok=True)

    # Retrieve session token
    response_tok = requests.get(TOKEN_URL).json()
    if response_tok["response_code"] != 0:
        raise RuntimeError(f"Token request failed: {response_tok}")
    session_token = response_tok["token"]

    # Update API URL
    API_URL = BASE_API_URL + "&token=" + session_token

    # Retrieve database
    while(True):
        response_qst = requests.get(API_URL).json()
        response_code = response_qst["response_code"]

        # Success Returned results successfully.
        if response_code == 0:
            questions.extend(q for q in response_qst["results"])
            print(f"Added {len(response_qst['results'])} questions (total: {len(questions)})")
            time.sleep(time_limit)
            continue

        # Token Empty Session Token has returned all possible questions 
        # for the specified query. Resetting the Token is necessary.
        if response_code == 4:
            print(f"\n==== Retrieval completed ====")
            print(f"Total number of questions: {len(questions)}")
            print()
            break

        # Rate Limit: Too many requests have occurred. 
        #  IP can only access the API once every 5 seconds.
        if response_code == 5:
            time.sleep(1)
            continue

    # Save database
    db = {"questions" : questions}
    json_db = json.dumps(db)
    with open(os.path.join(save_path, "trivia_db.json"), 'w') as f:
        f.write(json_db)

    print(f"=== FILE SAVED ===")
    print(f"File saved in path: {os.path.join(save_path, 'trivia_db.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path")
    args = parser.parse_args()

    try:
        retrieve_trivia_db(args.save_path)
    except Exception as e:
        print(f"Error during database retrieval: {e}")