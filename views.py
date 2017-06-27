import scipy.io
import scipy.stats
import datetime
import os
import numpy as np
import pandas as pd
import psycopg2
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import render_template
from flask import request
from app import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from flask_sqlalchemy import SQLAlchemy

from bokeh.embed import components
from bokeh.io import push_notebook, show, output_notebook
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.charts import Donut, Bar
from bokeh.layouts import row, gridplot
from bokeh.models import HoverTool


# username = 'HindSight'
# host = 'localhost'
# dbname = 'DonorPatient'
# password='HindSight'
# engine = create_engine('postgresql+psycopg2://%s:%s@13.58.100.1/%s'%(username,password,dbname))
# con = None
# con = psycopg2.connect(database = dbname, user = username, password = password)

X_df = pd.read_csv('./20170625DonorPatientPre_features.csv',index_col=0)
Y_df = pd.read_csv('./20170625DonorPatientPre_results.csv',index_col=0)
length = len(X_df)

@app.route('/')
@app.route('/index')
def index():
    return render_template('master.html')

@app.route('/go')
def go():
    query = request.args.get("query", '')
    query = query.encode('ascii','replace')
    query_proba = Y_df['Probability'].loc[query]
    microbial_composition=X_df.loc[query, X_df.columns != "AlphaDiversity"]

    Donor_composition_df=microbial_composition.to_frame()

    figure = make_plot(query)
    fig_script, fig_div = components(figure)
    figure_donor = make_plot_donor(X_df,query)
    fig_script_donor, fig_div_donor = components(figure_donor)
    if query_proba > 0.64:
    	query_results="Success!"
    else:
    	query_results="Failure."

    return render_template(
        'go.html',
        query_results=query_results, donorID=query, query_proba=str(query_proba),
        dataframe=Donor_composition_df.to_html(), 
        fig_script=fig_script, fig_div=fig_div,
        fig_script_donor=fig_script_donor, fig_div_donor=fig_div_donor
    )
# def stacked(df,categories):
# 	areas = dict()
# 	last = np.zeros(len(df[categories[0]]))
# 	for cat in categories:
# 		next = last + df[cat]
# 		areas[cat] = np.hstack((last[::-1],next))
# 		last = next
# 	return areas


def make_plot(query):
	# PercentageMicrobe_Success=X_df[Y_df['Results'] ==1 & X_df.columns != "AlphaDiversity"]].sum(axis=1)
    averageMicrobe_Success = X_df.ix[Y_df['Results'] ==1, X_df.columns != "AlphaDiversity"].mean(axis=0)
    averageMicrobe_Failure = X_df.ix[Y_df['Results'] ==0, X_df.columns != "AlphaDiversity"].mean(axis=0)
    averageMicrobe_Success.sort_values(ascending=False, inplace=True)
    averageMicrobe_Failure.sort_values(ascending=False, inplace=True)
    donor = X_df.ix[query, X_df.columns != "AlphaDiversity"].sort_values(ascending=False).to_frame()
    success_donor_Failure = pd.concat([averageMicrobe_Success, donor, averageMicrobe_Failure], axis=1, join_axes=[averageMicrobe_Success.index])
    success_donor_Failure.columns = ['Success', 'donor','Failure']
    colors = []
    for name, hex in matplotlib.colors.cnames.items():
        colors.append(name)
    pie_chart_success = Donut(success_donor_Failure['Success'][0:8]
        , title="Average successful donors' gut microbe composition"
        , palette=colors[8:16])
    pie_chart_failure = Donut(success_donor_Failure['Failure'][0:8]
        , title="Average failed donors"
        , palette=colors[8:16])
    pie_chart_donor   = Donut(success_donor_Failure['donor'][0:8]
        , title = 'The chosen donor'
        , palette=colors[8:16])
    p = row(pie_chart_success, pie_chart_donor, pie_chart_failure)
    return p

def make_plot_donor(X_df,query):
    colors=[]
    p = figure(
        plot_width = 50, 
        plot_height = 50, 
        responsive = True,
        title = "Mouse over the dots to see bacteria phyla detail")
    p.axis.visible = False
    p.outline_line_color = "black"
    # Set location of bubbles randomly scattered
    x = np.random.random(size=90) *300
    y = np.random.random(size=90) *300
    # Set log2(counts in each phyla) as radius, didn't use % composition because negative value
    # All samples are downsampled to 90000, so absolute value is comparable to % composition 
    percentcomp = list(X_df.ix[query, X_df.columns != "AlphaDiversity"].values)
    radius = np.log2(list(X_df.ix[query, X_df.columns != "AlphaDiversity"].values*90000+1))
    phylaNames = list(X_df.columns[X_df.columns != "AlphaDiversity"].values)
    source = ColumnDataSource(
        data = dict(
            desc = phylaNames, 
            percent = percentcomp ))

    for name, hex in matplotlib.colors.cnames.items():
        colors.append(name)

    # add a Circle renderer to this figure
    cr = p.circle(x, y, radius=radius,alpha=0.5, fill_color=colors[0:90],line_color=None, source = source)
    hover= HoverTool(tooltips="@desc", renderers = [cr])
    p.add_tools(hover)
    show(p)
    return p
