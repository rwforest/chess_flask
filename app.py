import os
import re
import logging
from flask import Flask, request, jsonify, send_from_directory # Import request
from flask_cors import CORS
# Assuming mlflow and SpanType are available in your environment
import mlflow
from mlflow.entities import SpanType
from openai import OpenAI

mlflow.openai.autolog()

# Define the base URL for the OpenAI client
base_url = 'https://8333330282859393.13.azuredatabricks.net/serving-endpoints'

# Initialize the OpenAI client globally (or within a function/class scope if preferred)
# Ensure the API key environment variable is set
DATABRICKS_TOKEN = ''

client = OpenAI(
    base_url=base_url,
    api_key=DATABRICKS_TOKEN, # Ensure os is imported
)

# Configure logging for Flask to suppress default messages
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

STATIC_FOLDER_PATH = os.path.join(os.path.abspath('.'), 'static')

# Initialize Flask, pointing to the static folder
flask_app = Flask(__name__, static_folder=STATIC_FOLDER_PATH)

CORS(flask_app, origins=["http://localhost:3000"])

# Define the route for the root URL
@flask_app.route('/api/best_move', methods=['POST'])
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
