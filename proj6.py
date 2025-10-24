import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import sqlite3
import json
import time
import os
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random
import openai

# --- Configuration ---
try:
    # Import your API key file
    import apikeyW
    openai.api_key = apikeyW.OPENAI_API_KEY
except ImportError:
    messagebox.showerror(
        "Error",
        "Could not import 'apikeyW.py'.\n"
        "Please create an 'apikeyW.py' file with your API key as:\n"
        "OPENAI_API_KEY = 'your_key_here'"
    )
    exit()
except AttributeError:
    messagebox.showerror(
        "Error",
        "'OPENAI_API_KEY' not found in 'apikeyW.py'.\n"
        "Please make sure the variable is named correctly."
    )
    exit()

# Model to use for analysis
OPENAI_MODEL = "gpt-4o-mini"
DB_FILE = 'feedback.db'

# --- Dummy Database Creation (Copied from original script) ---
DUMMY_REVIEWS = [
    "The display is absolutely breathtaking. It's like having a 4K TV for each eye. But, it's just too heavy. I couldn't wear it for more than 30 minutes without my neck hurting. The passthrough is magic, though.",
    "Mind-blowing! The eye tracking is flawless and feels like a superpower. I just wish there were more apps. Right now, it feels like an amazing tech demo. The price is also a major factor. Really steep.",
    "I'm returning it. The comfort is a deal-breaker. It's front-heavy and presses on my cheeks. Also, the battery life is terrible. I barely got 90 minutes. For this price, I expected perfection.",
    "This is the future of computing, period. The spatial video feature is emotionally powerful. I watched videos of my kids and it felt like I was back in that moment. Yes, it's expensive, but it's a gen-1 product. The potential is limitless.",
    "It's... okay? The 'wow' factor is huge at first. But the field of view is more limited than I expected, and the glare on the lenses is annoying. Persona is just creepy. Not worth the money for me.",
    "I've been using it for work, and it's a game-changer. Multiple massive, crystal-clear screens anywhere I want? Yes, please. The virtual keyboard is slow, but paired with a real one, it's my new office.",
    "The audio is surprisingly good! Very spatial. But the software feels a bit buggy and unfinished. Apps crash sometimes, and the 'VisionOS' store is empty. Holding out for version 2.",
    "I'm a developer, and the potential here is insane. The integration with the Apple ecosystem is seamless. But the device is isolating. My family says I look like a robot. It's a strange social experience.",
    "Price is way too high. For $3500, I expect it to do more than my laptop or TV, and it doesn't, really. It's a very cool, very expensive toy.",
    "The passthrough is the real star. It's almost zero-latency. But it's still grainy in low light. The Persona avatars are uncanny valley and need a lot of work. Good start, but not a must-have yet."
]

class SentimentApp:
    def __init__(self, master):
        self.master = master
        master.title("Apple Vision Pro Feedback Analysis")
        master.geometry("1000x800")

        # --- Main Layout ---
        self.main_frame = ttk.Frame(master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top: Controls ---
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Analysis",
            command=self.start_analysis_thread
        )
        self.start_button.pack(pady=10)

        self.progress_bar = ttk.Progressbar(
            self.control_frame,
            orient="horizontal",
            mode="indeterminate"
        )
        self.progress_bar.pack(fill=tk.X, pady=5)

        # --- Middle: Results ---
        self.results_frame = ttk.Frame(self.main_frame)
        self.results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.columnconfigure(1, weight=1)
        self.results_frame.rowconfigure(1, weight=1)
        self.results_frame.rowconfigure(3, weight=1)

        # Text Results
        self.sentiment_label = ttk.Label(self.results_frame, text="Overall Sentiment:", font=("Arial", 14, "bold"))
        self.sentiment_label.grid(row=0, column=0, sticky="w", pady=5)
        self.sentiment_text = ttk.Label(self.results_frame, text="...")
        self.sentiment_text.grid(row=0, column=1, sticky="w", pady=5)
        
        self.pos_aspect_label = ttk.Label(self.results_frame, text="Top Positive Aspects:", font=("Arial", 14, "bold"))
        self.pos_aspect_label.grid(row=2, column=0, sticky="w", pady=5)
        self.pos_aspect_text = ttk.Label(self.results_frame, text="...")
        self.pos_aspect_text.grid(row=2, column=1, sticky="w", pady=5)
        
        self.neg_aspect_label = ttk.Label(self.results_frame, text="Top Negative Aspects:", font=("Arial", 14, "bold"))
        self.neg_aspect_label.grid(row=4, column=0, sticky="w", pady=5)
        self.neg_aspect_text = ttk.Label(self.results_frame, text="...")
        self.neg_aspect_text.grid(row=4, column=1, sticky="w", pady=5)

        # Image Results
        self.sentiment_chart_label = ttk.Label(self.results_frame, text="Sentiment Distribution")
        self.sentiment_chart_label.grid(row=1, column=0, pady=5)
        self.sentiment_img_label = ttk.Label(self.results_frame, borderwidth=2, relief="sunken")
        self.sentiment_img_label.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        self.pos_wc_label = ttk.Label(self.results_frame, text="Positive Word Cloud")
        self.pos_wc_label.grid(row=3, column=0, pady=5)
        self.pos_wc_img_label = ttk.Label(self.results_frame, borderwidth=2, relief="sunken")
        self.pos_wc_img_label.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)

        self.neg_wc_label = ttk.Label(self.results_frame, text="Negative Word Cloud")
        self.neg_wc_label.grid(row=5, column=0, pady=5)
        self.neg_wc_img_label = ttk.Label(self.results_frame, borderwidth=2, relief="sunken")
        self.neg_wc_img_label.grid(row=5, column=1, sticky="nsew", padx=5, pady=5)

        # --- Bottom: Log ---
        self.log_frame = ttk.Frame(self.main_frame)
        self.log_frame.pack(fill=tk.BOTH, expand=True, side=tk.BOTTOM)
        
        self.log_label = ttk.Label(self.log_frame, text="Analysis Log:")
        self.log_label.pack(anchor="w")
        
        self.log_text = scrolledtext.ScrolledText(self.log_frame, height=10, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        """Adds a message to the log text widget on the main thread."""
        def _log():
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state='disabled')
        # Schedule the UI update on the main thread
        self.master.after(0, _log)

    def start_analysis_thread(self):
        """Starts the analysis in a new thread to avoid freezing the GUI."""
        self.start_button.config(state='disabled')
        self.progress_bar.start()
        
        # Clear previous results
        self.sentiment_text.config(text="...")
        self.pos_aspect_text.config(text="...")
        self.neg_aspect_text.config(text="...")
        self.sentiment_img_label.config(image='')
        self.pos_wc_img_label.config(image='')
        self.neg_wc_img_label.config(image='')

        # Start the background thread
        analysis_thread = threading.Thread(target=self.run_analysis, daemon=True)
        analysis_thread.start()

    def run_analysis(self):
        """The main analysis logic. Runs in a background thread."""
        try:
            self.log("Starting Apple Vision Pro Feedback Analysis...")
            self.create_dummy_database()
            reviews = self.get_reviews_from_db()

            if not reviews:
                self.log("No reviews found in the database.")
                return

            sentiments = []
            all_aspects = []
            
            self.log(f"Processing {len(reviews)} reviews...")
            
            for i, (review_id, review_text) in enumerate(reviews):
                self.log(f"  - Analyzing review {review_id} ({i+1}/{len(reviews)})...")
                
                sentiment = self.analyze_sentiment_openai(review_text)
                sentiments.append(sentiment)
                
                aspects = self.extract_aspects_openai(review_text)
                all_aspects.extend(aspects)
                
                time.sleep(1) # Be nice to the API

            # Run final analysis and visualization
            analysis_results = self.analyze_and_visualize(sentiments, all_aspects)
            
            # Send results to the main thread for GUI update
            self.master.after(0, self.update_gui, analysis_results)

        except Exception as e:
            self.log(f"An error occurred during analysis: {e}")
            messagebox.showerror("Analysis Error", f"An error occurred: {e}")
            self.master.after(0, self.analysis_finished)
            
    def update_gui(self, results):
        """Updates the GUI with the final results. Runs on the main thread."""
        self.log("Analysis complete. Updating GUI...")
        
        # Update text labels
        self.sentiment_text.config(text=results.get("sentiment_summary", "..."))
        self.pos_aspect_text.config(text=results.get("pos_summary", "..."))
        self.neg_aspect_text.config(text=results.get("neg_summary", "..."))

        # Update images
        self.load_image(results.get("sentiment_plot_path"), self.sentiment_img_label)
        self.load_image(results.get("pos_wc_path"), self.pos_wc_img_label)
        self.load_image(results.get("neg_wc_path"), self.neg_wc_img_label)

        self.analysis_finished()

    def analysis_finished(self):
        """Called when analysis is done or has failed."""
        self.progress_bar.stop()
        self.start_button.config(state='normal')
        
    def load_image(self, path, label_widget):
        """Loads an image file, resizes it, and displays it in a label."""
        if not path or not os.path.exists(path):
            self.log(f"Error: Image path not found: {path}")
            return

        try:
            # Get widget size
            label_widget.update_idletasks()
            width = label_widget.winfo_width()
            height = label_widget.winfo_height()

            # Ensure minimum size for initial load
            if width < 50: width = 400
            if height < 50: height = 300

            img = Image.open(path)
            img.thumbnail((width - 10, height - 10), Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img)
            
            # Store a reference to avoid garbage collection
            label_widget.image = img_tk 
            label_widget.config(image=img_tk)
            
        except Exception as e:
            self.log(f"Error loading image {path}: {e}")

    # --- Analysis Functions (Refactored as methods) ---

    def create_dummy_database(self):
        """Creates and populates a dummy SQLite database if it doesn't exist."""
        if os.path.exists(DB_FILE):
            self.log(f"'{DB_FILE}' already exists. Skipping creation.")
            return

        self.log(f"Creating dummy database '{DB_FILE}'...")
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT
        )
        ''')
        all_reviews = []
        for _ in range(4): # 21 * 4 = 84 reviews
            all_reviews.extend(DUMMY_REVIEWS)
        random.shuffle(all_reviews)
        for review in all_reviews:
            cursor.execute("INSERT INTO reviews (review_text) VALUES (?)", (review,))
        conn.commit()
        conn.close()
        self.log(f"Successfully populated '{DB_FILE}' with {len(all_reviews)} reviews.")

    def get_reviews_from_db(self):
        """Fetches all reviews from the SQLite database."""
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT id, review_text FROM reviews")
        reviews = cursor.fetchall()
        conn.close()
        return reviews

    def call_openai_api(self, system_prompt, user_prompt):
        """Reusable function to call the OpenAI ChatCompletions API."""
        delay = 1
        for _ in range(5):
            try:
                client = openai.OpenAI(api_key=openai.api_key)
                response = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                json_string = response.choices[0].message.content
                return json.loads(json_string)
            except openai.RateLimitError as e:
                self.log(f"Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except (openai.APIError, openai.APIConnectionError) as e:
                self.log(f"API Error: {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except json.JSONDecodeError as e:
                self.log(f"Error decoding JSON from API: {e}")
                self.log(f"Raw response: {response.choices[0].message.content}")
                return None
        self.log("Failed to call OpenAI API after several retries.")
        return None

    def analyze_sentiment_openai(self, review_text):
        """Analyzes the overall sentiment of a review using OpenAI."""
        system_prompt = (
            "You are a sentiment analysis expert. Analyze the following customer "
            "review for the Apple Vision Pro. Respond with ONLY a JSON object "
            "containing the sentiment. "
            "The JSON schema should be: "
            "{'sentiment': 'Positive' | 'Negative' | 'Neutral'}"
        )
        user_prompt = f"Review:\n\"\"\"\n{review_text}\n\"\"\""
        parsed_response = self.call_openai_api(system_prompt, user_prompt)
        if parsed_response and 'sentiment' in parsed_response:
            return parsed_response['sentiment']
        return "Neutral"

    def extract_aspects_openai(self, review_text):
        """Extracts specific aspects and their sentiments using OpenAI."""
        system_prompt = (
            "You are an expert product review analyst. Your task is to extract "
            "specific features/aspects (e.g., 'comfort', 'display', 'price', "
            "'apps', 'passthrough') mentioned in the review and identify the "
            "sentiment (Positive, Negative, Neutral) for each. Also, provide a "
            "brief quote from the review that supports your finding."
            "Respond with ONLY a JSON object containing a list of aspects."
            "The JSON schema should be: "
            "{'aspects': [{'aspect': 'feature_name', 'sentiment': 'Positive' | 'Negative' | 'Neutral', 'quote': 'supporting_quote'}]}"
        )
        user_prompt = f"Analyze this review:\n\"\"\"\n{review_text}\n\"\"\""
        parsed_response = self.call_openai_api(system_prompt, user_prompt)
        if parsed_response and 'aspects' in parsed_response and isinstance(parsed_response['aspects'], list):
            cleaned_aspects = []
            for item in parsed_response['aspects']:
                if 'aspect' in item:
                    item['aspect'] = item['aspect'].lower().strip().replace(" ", "_")
                    cleaned_aspects.append(item)
            return cleaned_aspects
        return []

    def plot_sentiment_distribution(self, sentiment_counts, filename='sentiment_distribution.png'):
        """Creates and saves a bar chart, returning the filename."""
        labels = sentiment_counts.keys()
        sizes = sentiment_counts.values()
        colors = ['#4CAF50', '#F44336', '#FFC107']
        
        plt.figure(figsize=(8, 6))
        plt.bar(labels, sizes, color=colors)
        plt.title('Overall Customer Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.savefig(filename)
        plt.close() # Close the plot to free memory
        self.log(f"Sentiment distribution chart saved to '{filename}'")
        return filename

    def create_word_cloud(self, aspect_list, filename='aspect_word_cloud.png'):
        """Creates and saves a word cloud, returning the filename."""
        if not aspect_list:
            self.log(f"No data to generate word cloud for '{filename}'.")
            return None
        
        text = ' '.join(aspect_list)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(filename)
        plt.close() # Close the plot to free memory
        self.log(f"Word cloud saved to '{filename}'")
        return filename

    def analyze_and_visualize(self, sentiments, all_aspects):
        """Runs the final analysis and creates visualizations, returning paths and summaries."""
        self.log("\n--- Analysis Complete ---")
        results = {}

        # 1. Sentiment Distribution
        sentiment_counts = Counter(sentiments)
        total_reviews = len(sentiments)
        sentiment_summary = []
        for sentiment, count in sentiment_counts.items():
            sentiment_summary.append(f"{sentiment}: {count} ({count/total_reviews:.1%})")
        results["sentiment_summary"] = " | ".join(sentiment_summary)
        self.log(f"Overall Sentiment: {results['sentiment_summary']}")
        results["sentiment_plot_path"] = self.plot_sentiment_distribution(sentiment_counts)

        # 2. Aspect Frequency and Sentiment
        positive_aspects = [a['aspect'] for a in all_aspects if a['sentiment'] == 'Positive']
        negative_aspects = [a['aspect'] for a in all_aspects if a['sentiment'] == 'Negative']
        
        positive_counts = Counter(positive_aspects)
        negative_counts = Counter(negative_aspects)
        
        # 3. Create Word Clouds
        results["pos_wc_path"] = self.create_word_cloud(positive_aspects, 'positive_aspects.png')
        results["neg_wc_path"] = self.create_word_cloud(negative_aspects, 'negative_aspects.png')
        
        # 4. Get Top Aspects
        pos_summary = ", ".join([f"{a} ({c})" for a, c in positive_counts.most_common(5)])
        results["pos_summary"] = pos_summary if pos_summary else "None"
        self.log(f"Top 5 Positive Aspects: {results['pos_summary']}")

        neg_summary = ", ".join([f"{a} ({c})" for a, c in negative_counts.most_common(5)])
        results["neg_summary"] = neg_summary if neg_summary else "None"
        self.log(f"Top 5 Negative Aspects: {results['neg_summary']}")
        
        return results

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()