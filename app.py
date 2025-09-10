from flask import Flask, request, render_template
from predict_client import predict_line

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.form["text"]
    try:
        category, doctors = predict_line(user_input)
        if doctors:
            doctors_str = ", ".join(doctors)
        else:
            doctors_str = "No doctors found"
        result = f"Category: {category} | Recommended Doctors: {doctors_str}"
    except Exception as e:
        result = f"Error: {e}"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
