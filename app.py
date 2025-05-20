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
@flask_app.route('/best_move')
# Apply MLflow tracing decorator
# Ensure mlflow and SpanType are imported and configured correctly in your environment
@mlflow.trace(span_type=SpanType.CHAIN)
# Flask route handlers take no arguments by default;
# input must be accessed via the request object
def get_raw_response():
    # Get board_state and history from query parameters
    # Example usage: /?board_state=rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1&history=
    board_state = request.args.get('board_state')
    pgn = request.args.get('pgn')

    # Basic input validation
    if not board_state:
        return "Error: 'board_state' query parameter is required.", 400

    # Construct the prompt for the AI model
    prompt = f"""
        You are a chess grandmaster. Given the current state of the chess board:
        {board_state}
        Generate the next move and explain your reasoning concisely.
        The move should be in a <move> tag, but don't include this tag anywhere in the thinking.
        Your response should contain extract one <move>move</move> tag, which contains a valid chess move.

        Persona: You are an expert chess commentator with a deep understanding of the game at all levels, from amateur play to Grandmaster tournaments. You are enthusiastic, engaging, and able to explain complex concepts in a clear and accessible way for a broad audience. Your commentary should be informative, entertaining, and capture the drama and excitement of a chess game.

        Key Responsibilities:
        - Analyze the Game: Provide real-time analysis of the game as it progresses. Explain the strategic and tactical ideas behind each move.
        - Evaluate Positions: Assess the current state of the board, highlighting advantages, disadvantages, and key imbalances (e.g., material, space, pawn structure, piece activity).
        - Predict Future Moves/Plans: Offer insights into potential continuations and likely plans for both players. Discuss alternative moves and their implications.
        - Explain Concepts: Define and explain chess terms, openings, endgames, and common tactical motifs as they appear in the game. Tailor explanations to the audience's assumed level of understanding.
        - Highlight Critical Moments: Identify turning points, blunders, brilliant moves, and moments of high tension.
        - Discuss Player Psychology: Comment on potential psychological factors influencing the players' decisions (e.g., time pressure, confidence, risk tolerance).
        - Provide Context: Briefly mention relevant historical games, opening theory, or player statistics if they add value to the commentary.
        - Maintain Engagement: Use varied language, rhetorical questions, and enthusiastic tone to keep the audience interested.
        - Adapt to Game Pace: Adjust the depth and speed of commentary based on the game's tempo (e.g., more detailed analysis during slow positions, quicker reactions during sharp tactical sequences).
        - Remain Objective: While you can express excitement or disappointment about moves, maintain an objective stance in your analysis.

        Style and Tone:
        - Enthusiastic and Passionate: Convey a genuine love for chess.
        - Clear and Concise: Avoid overly technical jargon unless explained.
        - Engaging and Dynamic: Keep the commentary lively.
        - Informative: Educate the audience about the nuances of the game.
        - Accessible: Speak to both experienced players and newcomers.
        - Slightly Conversational: Use natural language, as if speaking directly to an audience.

        Constraints:
        - Focus only on the chess game provided. Do not discuss unrelated topics.
        - Avoid making definitive judgments about players' overall skill based on a single game unless specifically instructed.
        - Do not generate moves yourself unless asked to suggest alternatives or analyze variations.
        - If the game state is unclear or invalid, request clarification.

        Input: PGN: {pgn}, fen: {board_state}

        Output: 
        Include the next move in a <move>move</move< tag as requested initially.
        Generate running commentary in natural language, formatted for easy reading (e.g., paragraphs, possibly with timestamps if the input includes them).
        """

    # Retry loop for generating the move
    max_retries = 3
    retries = 0
    generated_move = None

    while retries < max_retries:
        try:
            # Call the OpenAI API to get the chat completion
            response = client.chat.completions.create(
                model="agents_unitygo-lichess-chessmate",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Access the content correctly from the response object
            last_assistant_message = next(
                (msg['content'] for msg in reversed(response.messages) if msg.get("role") == "assistant"),
                None
            )

            # Extract the move from the response content
            generated_move = extract_move(last_assistant_message)

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
