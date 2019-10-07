from flask import Flask, render_template, flash
from Content_management import Content
TOPIC_DICT = Content()
import pygal
import sys

TOPIC_DICT = Content()

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("main.html")


@app.errorhandler(404)
def page_not_found(e):
	return render_template("404.html")

	

@app.route('/pygalexample/')
def pygalexample():
	try:
		graph = pygal.Line()
		graph.title = '% Change Coolness of programming languages over time.'
		graph.x_labels = ['2011','2012','2013','2014','2015','2016']
		graph.add('Python',  [15, 31, 89, 200, 356, 900])
		graph.add('Java',    [15, 45, 76, 80,  91,  95])
		graph.add('C++',     [5,  51, 54, 102, 150, 201])
		graph.add('All others combined!',  [5, 15, 21, 55, 92, 105])
		graph_data = graph.render_data_uri()
		graph2 = pygal.Bar()
		graph2.title = '% Some random notes.'
		graph2.x_labels = ['6eme','5eme','4eme','3eme','2nd','1ere', 'tle']
		graph2.add('Paul',  [14, 12.5, 13, 9, 8.45, 9, 11])
		graph2.add('Pascal',    [20, 17, 14, 11, 8, 10, 20])
		graph2.add('André',     [9, 12, 11, 14, 14, 10, 13])
		graph_data2 = graph2.render_data_uri()
		return render_template("graphing.html", graph_data = graph_data, graph_data2 = graph_data2 )
	except Exception:
		return(str(e))
		
if __name__ == "__main__":
    app.run()
