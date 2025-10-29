# Friday-Project-6

This is a Python GUI application built with Tkinter that analyzes customer feedback for the Apple Vision Pro. It fetches reviews from a local SQLite database, uses the OpenAI API (gpt-4o-mini) to perform sentiment and aspect-based analysis, and then displays an aggregated summary with charts and word clouds.

📊 Features
Simple GUI: A clean user interface built with tkinter.

AI-Powered Analysis: Uses OpenAI's gpt-4o-mini to analyze reviews for:

Overall Sentiment: Classifies each review as "Positive," "Negative," or "Neutral."

Aspect Extraction: Identifies specific product features (e.g., "comfort," "display," "price," "apps") and the sentiment associated with each.

Dynamic Visualizations: Generates and displays:

A bar chart of the overall sentiment distribution.

A word cloud of the most frequent positive aspects.

A word cloud of the most frequent negative aspects.

Responsive UI: Performs all API calls and analysis in a separate threading thread to prevent the GUI from freezing.

Dummy Data: Automatically creates a feedback.db SQLite database and populates it with sample reviews on the first run.

Live Logging: Shows the analysis progress in a scrolled log window.

⚙️ How It Works
Database Setup: On launch, the script checks for feedback.db. If it doesn't exist, it creates it and populates it with 40 sample Apple Vision Pro reviews.

Start Analysis: When the user clicks "Start Analysis," the app disables the button, starts a progress bar, and launches a new thread.

Background Processing: In this thread, the app: a. Fetches all reviews from the feedback.db. b. Loops through each review text. c. Makes two OpenAI API calls for each review: i. One call to get the overall sentiment. ii. A second call to extract a list of aspects (e.g., {'aspect': 'price', 'sentiment': 'Negative'}).

Aggregation & Visualization: Once all reviews are processed, the app: a. Aggregates the sentiment data (e.g., 20 Positive, 15 Negative, 5 Neutral). b. Aggregates the positive and negative aspects. c. Uses matplotlib to save a bar chart (sentiment_distribution.png). d. Uses wordcloud to save two word clouds (positive_aspects.png and negative_aspects.png).

Update GUI: The main thread is signaled to update the GUI, loading the newly created images and text summaries into the appropriate labels.

🚀 Setup & Installation
Before you can run this script, you need to install the required Python libraries and set up your OpenAI API key.

1. Install Dependencies
You must have Python 3 installed. You can install the required libraries using pip:

Bash

pip install openai pillow matplotlib wordcloud
(Note: tkinter and sqlite3 are included in the standard Python library.)

2. Set Up Your API Key
This script requires an OpenAI API key to function.

In the same directory as your Python script, create a new file named apikeyW.py.

Open this new file and add the following line, replacing 'your_key_here' with your actual OpenAI API key:

Python

OPENAI_API_KEY = 'your_key_here'
Save the file. The main script will automatically import this key.

🖥️ How to Run
Ensure you have completed the Setup & Installation steps above.

Save the main script as a Python file (e.g., sentiment_app.py).

Run the script from your terminal:

Bash

python sentiment_app.py
The application window will open. Click the "Start Analysis" button.

Wait for the analysis to complete. You can monitor the "Analysis Log" at the bottom of the window.

Once finished, the progress bar will stop, and the charts and summaries will appear in the window.

Generated Files
This script will create the following files in the same directory:

feedback.db: The SQLite database containing the reviews.

sentiment_distribution.png: The bar chart of overall sentiment.

positive_aspects.png: The word cloud for positive aspects.

negative_aspects.png: The word cloud for negative aspects.