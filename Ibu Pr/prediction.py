from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pickle



data = pd.read_csv('used_cars.csv')

model_labelencoder = LabelEncoder()
fuel_labelencoder = LabelEncoder()

model_labelencoder.fit(data['Model'].unique())
fuel_labelencoder.fit(data['Fuel'].unique())

data['Model'] = model_labelencoder.transform(data['Model'])
data['Fuel'] = fuel_labelencoder.transform(data['Fuel'])

data['Age'] = 2023 - data['Year']

X = data.drop('Price', axis=1)
y = data['Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('poly.pkl', 'wb') as f:
    pickle.dump(poly, f)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    year = float(request.form['year'])
    model_name = request.form['model']
    kilometres = float(request.form['kilometres'])
    fuel = request.form['fuel']
    age = 2023 - year  

    model_name = model_labelencoder.transform([model_name])[0] if model_name in model_labelencoder.classes_ else 0
    fuel = fuel_labelencoder.transform([fuel])[0] if fuel in fuel_labelencoder.classes_ else 0

    df = pd.DataFrame([[year, model_name, kilometres, fuel, age]], columns=['Year', 'Model', 'Kilometres', 'Fuel', 'Age'])
    df_scaled = scaler.transform(df)

    with open('poly.pkl', 'rb') as f:
        poly = pickle.load(f)
    df_scaled_poly = poly.transform(df_scaled)

    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
        
    prediction = loaded_model.predict(df_scaled_poly)
    prediction = np.maximum(0, prediction[0])

    return f"The predicted price of the car is ${prediction:.2f}. <a href='/feedback'>Is this accurate?</a>"

@app.route('/feedback')




def feedback():
    return "Thank you for your feedback. We will use it to improve our model."

if __name__ == '__main__':
    app.run(debug=True)
