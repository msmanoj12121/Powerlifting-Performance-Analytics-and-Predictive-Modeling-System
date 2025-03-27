from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from urllib.parse import unquote
from fuzzywuzzy import process
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load dataset
DATASET_PATH = r"C:\Users\Asus\Desktop\projects\powerlifting_dataset_real_names.csv"
df = pd.read_csv(DATASET_PATH)

# Clean data
df.dropna(inplace=True)
if "Name" not in df.columns:
    raise ValueError("Dataset must contain a 'Name' column!")

# Add total lift column
df["Total_Lift"] = df["Squat_Best"] + df["Bench_Best"] + df["Deadlift_Best"]

# Model training
features = ["Age", "Weight_Class", "Total_Lift", "Wilks_Score"]
df = df.dropna(subset=features)
df["Weight_Class"] = df["Weight_Class"].astype(str).str.extract(r'(\d+)').astype(float)
X = df[features]
y = df["Wilks_Score"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# 📍 Homepage
@app.route("/")
def home():
    return render_template("index.html")

# 🔍 Live Search Suggestions
# 🔍 Fetch Search Suggestions
from fuzzywuzzy import process

from fuzzywuzzy import process

@app.route("/search_suggestions", methods=["GET"])
def search_suggestions():
    query = request.args.get("q", "").strip().lower()
    print(f"🔍 Search Query: {query}")  # Debugging

    if not query:
        return jsonify([])

    # Get a list of unique lifter names
    lifter_names = df["Name"].dropna().unique()

    # Use fuzzy matching to find the 5 most similar names
    results = process.extract(query, lifter_names, limit=5)
    
    # Extract names with a similarity score of at least 50
    suggestions = [name for name, score in results if score >= 50]  

    print(f"✅ Suggested Names: {suggestions}")  # Debugging
    return jsonify(suggestions)




# 🔎 Search Button Logic
@app.route("/search_lifter", methods=["GET"])
def search_lifter():
    query = request.args.get("query", "").strip().lower()
    print(f"Search button query: {query}")  # DEBUG

    if not query:
        return jsonify([])

    best_match, score = process.extractOne(query, df["Name"].dropna().unique())
    print(f"Best match found: {best_match} (Score: {score})")  # DEBUG

    if score >= 70:
        return jsonify({"redirect": f"/analyze_lifter/{best_match}"})
    
    return jsonify([])

# 🎯 Lifter Analysis Page
@app.route('/analyze_lifter/<lifter_name>')
def analyze_lifter(lifter_name):
    lifter_data = df[df['Name'] == lifter_name]  # Filter data for the player

    if lifter_data.empty:
        return render_template("player_details.html", error="Lifter not found!")

    lifter_overview = lifter_data.iloc[-1].to_dict()  # Latest record
    yearwise_stats = lifter_data[['Year', 'Total_Lift', 'Wilks_Score', 'Weight_Class']].to_dict(orient="records")

    return render_template("player_details.html", lifter=lifter_overview, yearwise_stats=yearwise_stats)


# 🏆 Top 3 Best Lifters
@app.route("/best_lifters", methods=["GET"])
def best_lifters():
    year = request.args.get("year", None)

    # Filter data by year if provided
    if year:
        filtered_df = df[df["Year"] == int(year)]
    else:
        filtered_df = df

    # Check if 'Gender' column exists in your dataset
    if "Gender" not in df.columns:
        raise ValueError("Dataset must contain a 'Gender' column!")

    # Top 3 overall lifters
    overall_lifters = filtered_df.nlargest(3, "Wilks_Score")[["Name","Total_Lift","Wilks_Score"]].to_dict(orient="records")

    # Top 3 male lifters
    male_lifters = filtered_df[filtered_df["Gender"].str.strip().str.lower() == "men"].nlargest(3, "Wilks_Score")[["Name","Total_Lift", "Wilks_Score"]].to_dict(orient="records")

    # Top 3 female lifters
    female_lifters = filtered_df[filtered_df["Gender"].str.strip().str.lower() == "women"].nlargest(3, "Wilks_Score")[["Name","Total_Lift", "Wilks_Score","Gender"]].to_dict(orient="records")

    # Available years for the dropdown filter
    years = sorted(df["Year"].unique(), reverse=True)

    return render_template("best_lifter.html", 
                           overall_lifters=overall_lifters,
                           male_lifters=male_lifters, 
                           female_lifters=female_lifters,
                           years=years,
                           selected_year=year)



# 📊 Predict Winner
@app.route("/predict_winner", methods=["GET"])
def predict_winner():
    X_scaled = scaler.transform(df[features])
    df["Predicted_Wilks_2026"] = model.predict(X_scaled)
    top_winners = df.nlargest(3, "Predicted_Wilks_2026")[["Name", "Predicted_Wilks_2026"]].to_dict(orient="records")
    return render_template("predict_winnner.html", winners=top_winners)

# 📅 Top Players by Year
@app.route("/top_players", methods=["GET"])
def top_players():
    years = sorted(df["Year"].unique(), reverse=True)
    selected_year = request.args.get("year")

    players = []
    if selected_year:
        selected_year = int(selected_year)

        # Ensure Federation column is included
        if "Federation" in df.columns:
            players = df[df["Year"] == selected_year][["Name", "Weight_Class", "Wilks_Score", "Total_Lift", "Federation"]]
        else:
            players = df[df["Year"] == selected_year][["Name", "Weight_Class", "Wilks_Score", "Total_Lift"]]
            players["Federation"] = "N/A"  # Fill missing Federation values

        # Sort players by Wilks Score
        players = players.sort_values(by="Wilks_Score", ascending=False).head(10).to_dict(orient="records")

    return render_template("top_players.html", years=years, selected_year=selected_year, players=players)


if __name__ == "__main__":
    app.run(debug=True)
