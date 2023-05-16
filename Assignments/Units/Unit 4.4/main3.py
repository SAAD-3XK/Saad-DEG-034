import json
import logging
import threading
import time
import os

from flask import Flask, render_template, request

app = Flask(__name__)


logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


TODO_FILE_NAME = "/home/app/vol/todo.json"
if os.path.exists(TODO_FILE_NAME):
    with open(TODO_FILE_NAME) as f:
        TODO_ITEMS = json.load(f)
else:
    TODO_ITEMS = []

def periodically_save_todo_items():
    while True:
        time.sleep(1)
        f=open(TODO_FILE_NAME, "w")
        json.dump(TODO_ITEMS, f)


saving_thread = threading.Thread(target=periodically_save_todo_items)
saving_thread.start()


@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "POST":
        content = request.form["content"]
        TODO_ITEMS.append(content)

    return render_template("index.html", todo_items=TODO_ITEMS)


if __name__ == "__main__":
    #this will run the app in debug mode or info mode according the ENV variable that is passed in docker-compose file
    if os.environ.get('LOG_LEVEL') == "info":
        app.debug = False
    else:
        app.debug = True
    app.run(host="0.0.0.0")
