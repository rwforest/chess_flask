import os
import re
import logging
import json # Added import for json
from flask import Flask, request, jsonify, send_from_directory # Import request
from flask_cors import CORS
# Assuming mlflow and SpanType are available in your environment
import mlflow
from mlflow.entities import SpanType
from openai import OpenAI
import chess

mlflow.openai.autolog()

# Define the base URL for the OpenAI client
base_url = "https://adb-8333330282859393.13.azuredatabricks.net/serving-endpoints"

# Initialize the OpenAI client globally (or within a function/class scope if preferred)
# Ensure the API key environment variable is set
DATABRICKS_TOKEN = ""
# Configure logging for Flask to suppress default messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

STATIC_FOLDER_PATH = os.path.join(os.path.abspath('.'), 'static')

# Initialize Flask, pointing to the static folder
flask_app = Flask(__name__, static_folder=STATIC_FOLDER_PATH)

CORS(flask_app, origins=["http://localhost:3000"])

client = OpenAI(
    base_url=base_url,
    api_key=DATABRICKS_TOKEN,
)

@flask_app.route("/api/llm_comments", methods=['POST'])
def analyze_position_llm():
    """
    Analyzes a chess position using the Gemini API (synchronous version).

    Args:
        fen: The FEN string representing the chess position.

    Returns:
        A list of analysis comments (dictionaries with 'speaker' and 'text')
        or an error message dictionary, or None if game is over.
    """

    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
        fen = data.get('fen', '8/1P1R4/n1r2B2/3Pp3/1k4P1/6K1/Bppr1P2/2q5 w - - 0 1')
        pgn = data.get('pgn', "") # pgn is fetched but not used in this function
    except Exception as e:
        error_response = {"error": str(e), "type": type(e).__name__}
        return jsonify(error_response), 500

    board = chess.Board(fen) 

    # Game over check was originally here, can be reinstated if needed
    # if game_over:
    #     print("Game is over. Skipping analysis.")  # Print to stderr
    #     return None

    print("Loading analysis...")  # Print to stderr
    analysis_comment = None

    prompt = (
        f"Analyze the chess position represented by this FEN: {fen}. "
        "Provide a brief, conversational analysis in three parts: "
        "one observation for White, one for Black, and a concluding summary from an Analyst. "
        "Format the response as a JSON array of objects, where each object has 'speaker' "
        "('White', 'Black', or 'Analyst') and 'text' (the analysis comment). "
        'Example: [{"speaker": "White", "text": "White controls the center."}, '
        '{"speaker": "Black", "text": "Black has a passed pawn on a5."}, '
        '{"speaker": "Analyst", "text": "The position is dynamically balanced."}]'
    )

    chat_history = [{"role": "user", "content": prompt}]

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "chess_commentary",
            "description": "Structured commentary for a chess game with identified speakers.",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "speaker": {
                            "type": "string",
                            "description": "The name or role of the person giving the commentary",
                        },
                        "text": {
                            "type": "string",
                            "description": "The commentary spoken by the speaker",
                        },
                    },
                    "required": ["speaker", "text"],
                },
            },
        },
    }

    try:
        # Using requests.post for a synchronous call
        response = client.chat.completions.create(
            model="databricks-llama-4-maverick",
            messages=chat_history,
            response_format=response_format,
        )

        analysis_comment = response.choices[0].message.model_dump()["content"]

    except (
        # openai.APIConnectionError, openai.APIStatusError, etc.
        # More specific OpenAI exceptions can be caught here if needed.
        # requests.exceptions.RequestException # This is not used when using OpenAI client library
    ) as e:  # Handles network errors, etc. for requests library
        print(f"Error fetching analysis from LLM: {e}")
        analysis_comment = [
            {
                "speaker": "Error",
                "text": f"Could not connect to analysis service: {str(e)}",
            }
        ]
    except Exception as e:  # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        analysis_comment = [
            {"speaker": "Error", "text": f"An unexpected error occurred: {str(e)}"}
        ]
    finally:
        print("Analysis loading complete.")
        return jsonify(json.loads(analysis_comment)), 200

# Define the route for the root URL
@flask_app.route('/api/best_move', methods=['GET'])
def get_move():
    fen = request.args.get('fen', '8/1P1R4/n1r2B2/3Pp3/1k4P1/6K1/Bppr1P2/2q5 w - - 0 1')
    pgn = request.args.get('pgn', "")

    # fen = '8/1P1R4/n1r2B2/3Pp3/1k4P1/6K1/Bppr1P2/2q5 w - - 0 1'
    # pgn = ""

    legal_moves = list(chess.Board(fen).legal_moves)  # This gives a list of Move objects
    legal_moves_str = [move.uci() for move in legal_moves]

    board_state = fen

    # Basic input validation
    if not board_state:
        return "Error: 'board_state' query parameter is required.", 400

    # Construct the prompt for the AI model
    prompt =  f"""You are a chess grandmaster. Given the current state of the chess board:
    {board_state}
    Legal moves: {legal_moves_str}
    Generate the next move and explain your reasoning concisely.
    The move should be in a <move> tag
    Make sure you only output one <move>move</move> in the response"""

    # Retry loop for generating the move
    max_retries = 3
    retries = 0
    generated_move = None

    while retries < max_retries:
        try:
            # Call the OpenAI API to get the chat completion
            response = client.chat.completions.create(
                model="databricks-claude-sonnet-4",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            print(response.choices[0].message.content)
            # Extract the move from the response content
            generated_move = extract_move(response.choices[0].message.content)

            # If a move was successfully extracted, break the loop
            if generated_move:
                break

        except Exception as e:
            # Log the error and retry
            logging.error(f"API call or processing failed: {e}")
            pass # Allow retry

        retries += 1

    # Return the extracted move or an error message
    if generated_move:
        data = {'move': generated_move}
        return jsonify(data), 200 # Return the move and a success status code
    else:
        return "Error: Could not extract a valid move from the AI response after multiple retries.", 500 # Return an error status code

@flask_app.route('/')
def serve_root():
    return send_from_directory(flask_app.static_folder, 'index.html')

@flask_app.route('/<path:path>')
def serve_static_assets(path):
    return send_from_directory(flask_app.static_folder, path)

# Define a static method to extract the move using regex
def extract_move(response_content):
    """Extracts move in <move> tags."""
    # Regex pattern to find the last occurrence of <move>...</move>
    pattern = re.compile(r'<move>(?!.*<move>)(.*?)</move>', re.DOTALL | re.IGNORECASE)
    match = pattern.search(response_content)
    if match:
        return match.group(1).strip()
    return None

# Run the Flask application if the script is executed directly
if __name__ == '__main__':
    # Ensure debug is True only in development
    flask_app.run(debug=True)
