import os
from flask import Flask, render_template

# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__, 
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Enterprise Sentiment Agent Web UI on port 5000...")
    app.run(debug=True, port=5000)
