from flask import Flask, render_template
# from flask.ext.sqlalchemy import SQLAlchemy

app = Flask(__name__)
# db = SQLAlchemy(app)

global_counter=0

@app.route("/")
def hello_world():
    # return render_template("index.html")


@app.route("/test")
def hello_world1():
    global global_counter
    global_counter+=1
    return f"<p>Hello,{global_counter} World!</p>"
    # return render_template("index.html")

if __name__ == '__main__':
	app.run(debug=True)