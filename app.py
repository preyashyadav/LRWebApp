from flask import Flask, render_template, request, flash, redirect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import os
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
# Create our own data
datafile = pd.DataFrame()
datafile[0] = np.arange(2000, 2020)
datafile[1] = [50, 54, 62, 78, 65, 69, 72, 77, 85, 98, 75, 80, 82, 85, 88, 75, 90, 93, 95, 98]
datafile.columns = ["Year", "Placement"]

regress = linear_model.LinearRegression()
regress.fit(datafile[['Year']], datafile[['Placement']])


def generate_plot(datafile, original_data):
    min_year = min(datafile['Year'].min(), original_data['Year'].min())
    max_year = max(datafile['Year'].max(), original_data['Year'].max())

    min_placement = min(datafile['Placement'].min(), original_data['Placement'].min())
    max_placement = max(datafile['Placement'].max(), original_data['Placement'].max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot with line of best fit
    axes[0].scatter(datafile['Year'], datafile['Placement'], label='Data Points')
    axes[0].set_title("Placement plot")
    axes[0].set_xlim(min_year, max_year)
    axes[0].set_ylim(min_placement, max_placement)
    axes[0].set_xticks(np.arange(min_year, max_year, 2))
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Placement")

    # Adding the line of best fit
    x_values = np.arange(min_year, max_year + 1)
    y_values = regress.predict(x_values.reshape(-1, 1))
    axes[0].plot(x_values, y_values, color='red', label='Line of Best Fit')
    axes[0].legend()

    # Correlation heatmap
    data_corr = datafile.corr()
    sb.heatmap(data_corr, annot=True, ax=axes[1])
    axes[1].set_title("Correlation Heatmap")

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return base64.b64encode(output.getvalue()).decode()



def perform_linear_regression(datafile):
    regress = linear_model.LinearRegression()
    regress.fit(datafile[['Year']], datafile[['Placement']])
    predicted_placement = regress.predict(datafile[['Year']])
    return [(year, placement[0]) for year, placement in zip(datafile['Year'], predicted_placement)]


def evaluate_regression_model(datafile):
    train_x, test_x, train_y, test_y = train_test_split(datafile[['Year']], datafile[['Placement']], test_size=0.30, random_state=42)
    regress = linear_model.LinearRegression()
    regress.fit(train_x, train_y)
    y_predicted = regress.predict(test_x)
    mae = metrics.mean_absolute_error(test_y, y_predicted)
    mse = metrics.mean_squared_error(test_y, y_predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(test_y, y_predicted)
    return mae, mse, rmse, r2, regress.coef_[0][0], regress.intercept_[0]



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/output', methods=['POST'])
def output():
    original_data = pd.DataFrame()
    original_data[0] = np.arange(2000, 2020)
    original_data[1] = [50, 54, 62, 78, 65, 69, 72, 77, 85, 98, 75, 80, 82, 85, 88, 75, 90, 93, 95, 98]
    original_data.columns = ["Year", "Placement"]

    if 'sample-data' in request.form:
        # Use sample data
        datafile = pd.DataFrame()
        datafile[0] = np.arange(2000, 2020)
        datafile[1] = [50, 54, 62, 78, 65, 69, 72, 77, 85, 98, 75, 80, 82, 85, 88, 75, 90, 93, 95, 98]
        datafile.columns = ["Year", "Placement"]
    elif 'file' in request.files:
        # Use uploaded data
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            datafile = pd.read_csv(file)
            datafile.columns = ["Year", "Placement"]
        else:
            flash('Invalid file format. Please upload a CSV file.')
            return redirect('/')
    else:
        flash('Invalid request.')
        return redirect('/')

    predicted_placement = perform_linear_regression(datafile)
    mae, mse, rmse, r2, coefficient, intercept = evaluate_regression_model(datafile)
    plot = generate_plot(datafile, original_data)  # Pass both datafile and original_data as arguments

    # Pass datafile to the template
    return render_template('output.html', predicted_placement=predicted_placement, mae=mae, mse=mse, rmse=rmse,
                           r2=r2, coefficient=coefficient, intercept=intercept, plot=plot, datafile=datafile)



if __name__ == '__main__':
    app.run(debug=True)
