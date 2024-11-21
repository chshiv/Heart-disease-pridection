from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Config for SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Create a User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create the database tables (run this once to initialize the DB)
with app.app_context():
    db.create_all()

# Load dataset
data = pd.read_csv('heart_disease.csv')

# X is the feature matrix, and y is the target (0 for no heart disease, 1 for heart disease)
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Fit models
model_accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    model_accuracies[model_name] = accuracy

# Save models and scaler
for model_name, model in models.items():
    with open(f'{model_name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# Routes

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Fetch the user from the database
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            session['logged_in'] = True
            session['email'] = user.email
            return redirect(url_for('symptom'))
        else:
            return "Invalid credentials. Please try again."
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return "Passwords do not match. Please try again."

        # Check if the user already exists
        user_exists = User.query.filter_by(email=email).first()
        if user_exists:
            return "User already exists. Please login."

        # Hash the password and create a new user using pbkdf2:sha256
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(email=email, password=hashed_password)

        # Add and commit the new user to the database
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('email', None)
    return redirect(url_for('home'))


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve all the inputs from the form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        resting_bp = int(request.form['resting_bp'])
        cholesterol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        max_heart_rate = int(request.form['thalach'])
        cp_type = int(request.form['cp'])
        major_vessels = int(request.form['ca'])
        thal = int(request.form['thal'])
        restecg = int(request.form['slope'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])

        # Create feature vector for prediction
        features = np.array([age, sex, resting_bp, cholesterol, fbs, max_heart_rate, cp_type, major_vessels, thal, restecg, exang, oldpeak]).reshape(1, -1)

        # Load scaler and standardize the features
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        features_scaled = scaler.transform(features)

        # Make predictions using all models
        predictions = {}
        for model_name in models.keys():
            with open(f'{model_name}_model.pkl', 'rb') as f:
                model = pickle.load(f)
            prediction = model.predict_proba(features_scaled)[0][1] * 100
            predictions[model_name] = {
                'prediction': round(prediction, 2),
                'accuracy': round(model_accuracies[model_name] * 100, 2)
            }

        # Render the result page with predictions from all models
        return render_template('result.html', predictions=predictions)


@app.route('/symptom', methods=['GET', 'POST'])
def symptom():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Retrieve user input from form
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        resting_bp = int(request.form['resting_bp'])
        cholesterol = int(request.form['cholesterol'])
        fbs = int(request.form['fbs'])
        max_heart_rate = int(request.form['max_heart_rate'])
        cp_type = int(request.form['cp_type'])
        major_vessels = int(request.form['major_vessels'])
        thal = int(request.form['thal'])

        # Create feature vector for prediction
        features = np.array([age, sex, resting_bp, cholesterol, fbs, max_heart_rate, cp_type, major_vessels, thal]).reshape(1, -1)

        # Load scaler and standardize the features
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        features_scaled = scaler.transform(features)

        # Make predictions using all models
        predictions = {}
        for model_name in models.keys():
            with open(f'{model_name}_model.pkl', 'rb') as f:
                model = pickle.load(f)
            prediction = model.predict_proba(features_scaled)[0][1] * 100
            predictions[model_name] = {
                'prediction': round(prediction, 2),
                'accuracy': round(model_accuracies[model_name] * 100, 2)
            }

        # Render the result page with predictions from all models
        return render_template('result.html', predictions=predictions)

    return render_template('symptom.html')


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        phone = request.form['phone']
        message = request.form['message']
        return f"Thank you {name}, we will get back to you soon."
    return render_template('getintouch.html')


if __name__ == '__main__':
    app.run(debug=True)
