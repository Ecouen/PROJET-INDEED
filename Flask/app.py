from __future__ import print_function
from flask import Flask, render_template, flash #Les modules pour pouvoir rendre notre code sur la page web
import pygal #Le module qui nous permet de faire les graphs
import pandas as pd
import numpy as np
from flask import request
import json #(WIP)Permet d'exporter des choses en format json
from flask import Flask #Attention à bien import tout les Flask differents
import flask
import random
from pygal.style import Style #Permet de configurer le style du 2ème module
from statistics import mean
from statistics import median
from pymongo import MongoClient



'''
DF TO MONGO

df
client=MongoClient()
db = client.Indeed
table = db.Flask
table
table.insert_many(df.to_dict(orient='record'))

'''


client = MongoClient()
db = client['Indeed']
#select the collection within the database
test = db.Flask
#convert entire collection to Pandas dataframe
df = pd.DataFrame(list(test.find()))



custom_style = Style(
  background='transparent',
  title_font_family = 'googlefont:Courier New',
  plot_background='transparent',
  foreground='#4f5f76',
  foreground_strong='#ffffad',
  foreground_subtle='#ffffff',
  value_label_font_size = 20,
  legend_font_size = 22,
  title_font_size = 18,
  tooltip_font_size = 30,
  value_colors = '#1561ad',
  opacity='.4',
  opacity_hover='.55',
  transition='500ms ease-in',
  colors=('#E853A0','#01d28e', '#985654', '#E87653', '#E8537A', '#1561ad'))



grph_slr_h = df.Salary[df.Salary_type == 'h']
grph_slr_m = df.Salary[df.Salary_type == 'm']
grph_slr_y = df.Salary[df.Salary_type == 'y']
grph_slr_w = df.Salary[df.Salary_type == 'w']
grph_slr_d = df.Salary[df.Salary_type == 'd']
grph_slr_tot = df["Salary"]
grph_slr_h = grph_slr_h.sample(20)
grph_slr_w = grph_slr_w.sample(20).tolist()
grph_slr_d = grph_slr_d.sample(20).tolist()
grph_slr_y = grph_slr_y.sample(20).tolist()
grph_slr_m = grph_slr_m.sample(20).tolist()
grph_slr_h = grph_slr_h.reset_index()
grph_slr_h = grph_slr_h.reset_index()
testas = grph_slr_h["level_0"]
grph_slr_h = grph_slr_h["Salary"].tolist()
Sal_loc = df.groupby(['Job_class', 'Location'])

Sal_loc_med = Sal_loc.median()
Sal_loc_med = Sal_loc_med.reset_index()
dev_med = Sal_loc_med[0:5]
data_med = Sal_loc_med[5:10]
sal_dev_med = dev_med["Salary"].tolist()
sal_data_med = data_med["Salary"].tolist()

loc = ["Toulouse", "Bordeaux", "Nantes", "Lyon", "Paris"]

Sal_loc_mean = Sal_loc.mean()
Sal_loc_mean = Sal_loc_mean.reset_index()
dev_mean = Sal_loc_mean[0:5]
data_mean = Sal_loc_mean[5:10]
sal_dev_mean = dev_mean["Salary"].tolist()
sal_data_mean = data_mean["Salary"].tolist()



Sal_mean_region_gb= df.groupby(['Location'])
Sal_mean_region = Sal_mean_region_gb.mean()
Sal_mean_region = Sal_mean_region.reset_index()
Sal_mean_region = Sal_mean_region["Salary"].tolist()
Sal_mean_region

Sal_median_region_gb= df.groupby(['Location'])
Sal_median_region = Sal_median_region_gb.median()
Sal_median_region = Sal_median_region.reset_index()
Sal_median_region = Sal_median_region["Salary"].tolist()
Sal_median_region


nb_exp= df.groupby(['Experience'])
nb_exp= nb_exp.count()
nb_exp = nb_exp["entreprise"].tolist()

nb2_exp= df.groupby(['Experience'])
mean_exp = nb2_exp.mean()
mean_salary_xp = mean_exp["Salary"].tolist()

med_exp = nb2_exp.median()
med_salary_xp = med_exp["Salary"].tolist()


nb_offre = len(df)
Moyenne = grph_slr_tot.mean().round(2) 

app = Flask(__name__)

@app.route('/')
def home():
    chart6 = pygal.Bar(style = custom_style)
    chart6.title = 'Moyenne des salaires par villes'
    chart6.x_labels = ["Moyenne"]
    chart6.add("Toulouse", Sal_mean_region[0])
    chart6.add("Bordeaux", Sal_mean_region[1])
    chart6.add("Nantes", Sal_mean_region[2])
    chart6.add("Lyon", Sal_mean_region[3])
    chart6.add("Paris", Sal_mean_region[4])
    graph_data6 = chart6.render_data_uri()
    return render_template("main.html", main_data = graph_data6, test = nb_offre, moyenne = Moyenne)



@app.errorhandler(404)
def page_not_found(e):
	return render_template("404.html")

@app.route('/Salaires/freq')
def Graphs_frq():
    chart2 = pygal.Bar(style = custom_style)
    chart2.title = 'Mediane des salaires pour chaque type de salaire'
    chart2.x_labels = ['Médiane des salaires']
    chart2.add("Hebdomadaire", median(grph_slr_w))
    chart2.add("Journalier", median(grph_slr_d))
    chart2.add("Mensuel", median(grph_slr_m) )
    chart2.add("Annuel", median(grph_slr_y) )
    chart2.add("Horaire", median(grph_slr_h) )
    graph_data3 = chart2.render_data_uri()


    chart3 = pygal.Bar(style = custom_style)
    chart3.title = 'Moyenne des salaires pour chaque type de salaire'
    chart3.x_labels = ['Moyenne des salaires']
    chart3.add("Hebdomadaire", mean(grph_slr_w))
    chart3.add("Journalier", mean(grph_slr_d))
    chart3.add("Mensuel", mean(grph_slr_m) )
    chart3.add("Annuel", mean(grph_slr_y) )
    chart3.add("Horaire", mean(grph_slr_h) )
    graph_data2 = chart3.render_data_uri()
    return render_template("Salaire/grph-freq.html", main_data2 = graph_data2, main_data3 = graph_data3,)



@app.route('/Salaires/region/OverallRegion')
def Graphs_reg():

    chart6 = pygal.Bar(style = custom_style)
    chart6.title = 'Moyenne des salaires par r&gion'
    chart6.x_labels = ["Moyenne"]
    chart6.add("Toulouse", Sal_mean_region[0])
    chart6.add("Bordeaux", Sal_mean_region[1])
    chart6.add("Nantes", Sal_mean_region[2])
    chart6.add("Lyon", Sal_mean_region[3])
    chart6.add("Paris", Sal_mean_region[4])
    graph_data6 = chart6.render_data_uri()

    chart7 = pygal.Bar(style = custom_style)
    chart7.title = 'Médianne des salaires par région'
    chart7.x_labels = ['Médianne']
    chart7.add("Toulouse", Sal_median_region[0])
    chart7.add("Bordeaux", Sal_median_region[1])
    chart7.add("Nantes", Sal_median_region[2])
    chart7.add("Lyon", Sal_median_region[3])
    chart7.add("Paris", Sal_median_region[4])
    graph_data7 = chart7.render_data_uri()
    return render_template("Salaire/Region/grph-reg-overall.html", main_data6 = graph_data6, main_data7 = graph_data7,)




@app.route('/Salaires/region/CatMetier')
def Graphs_ovrl_reg():

    chart4 = pygal.Bar(style = custom_style)
    chart4.title = 'Médianne des salaires correspondant a chaque catégorie de métier'
    chart4.x_labels = loc
    chart4.add("Salaire data", sal_data_med)
    chart4.add("Salaire dev", sal_dev_med)
    graph_data5 = chart4.render_data_uri()


    chart5 = pygal.Bar(style = custom_style)
    chart5.title = 'Moyenne des salaires correspondant a chaque catégorie de métier'
    chart5.x_labels = loc
    chart5.add("Salaire data", sal_data_mean)
    chart5.add("Salaire dev", sal_dev_mean)
    graph_data4 = chart5.render_data_uri()
    return render_template("Salaire/grph-reg-cat.html", main_data4 = graph_data4, main_data5 = graph_data5,)

@app.route('/Salaires/SalExp')
def Sal_per_exp():

    chart10 = pygal.Bar(style = custom_style)
    chart10.title = "Moyenne des salaires en fonction de l'éxperience"
    chart10.x_labels = ["-3 ans", "3 à 5 ans", "+5 ans"]
    chart10.add("Salaire Moyen", mean_salary_xp)
    graph_data10 = chart10.render_data_uri()


    chart9 = pygal.Bar(style = custom_style)
    chart9.title = "Médiane des salaires en fonction de l'éxperience"
    chart9.x_labels = ["-3 ans", "3 à 5 ans", "+5 ans"]
    chart9.add("Salaire median", med_salary_xp)
    graph_data9 = chart9.render_data_uri()
    return render_template("Salaire/Sal_XP.html", main_data9 = graph_data9, main_data10 = graph_data10,)



@app.route('/Salaires')
def testa():
    return render_template("Salaire.html")

@app.route('/Salaires/Salaires')
def testa2():
    return render_template("Salaire.html")

@app.route("/Other/OffresExp")
def Offxp():
    chart8 = pygal.Pie(half_pie=True, style = custom_style )
    chart8.title = "Nombre d'offres en fonction de l'experience"
    chart8.add('Junior', nb_exp[0])
    chart8.add('Sénior', nb_exp[1])
    chart8.add('Manager ou supérieur', nb_exp[2])
    graph_data8 =  chart8.render_data_uri()
    return render_template("Other/OffresExp.html", main_data8 = graph_data8)
    





if __name__ == "__main__":
    app.run()
