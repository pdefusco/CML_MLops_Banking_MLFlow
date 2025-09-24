#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/


import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import os
import cml.data_v1 as cmldata
import pyspark.pandas as ps

# SET USER VARIABLES
USERNAME = os.environ["PROJECT_OWNER"]
DBNAME = os.environ["DBNAME_PREFIX"]+"_"+USERNAME
CONNECTION_NAME = os.environ["SPARK_CONNECTION_NAME"]

# CREATE SPARK SESSION WITH DATA CONNECTIONS
conn = cmldata.get_connection(CONNECTION_NAME)
spark = conn.get_spark_session()

# READ LATEST ICEBERG METADATA
snapshot_id = spark.read.format("iceberg").load('{0}.transactions_{1}.snapshots'.format(DBNAME, USERNAME)).select("snapshot_id").tail(1)[0][0]
committed_at = spark.read.format("iceberg").load('{0}.transactions_{1}.snapshots'.format(DBNAME, USERNAME)).select("committed_at").tail(1)[0][0].strftime('%m/%d/%Y')
parent_id = spark.read.format("iceberg").load('{0}.transactions_{1}.snapshots'.format(DBNAME, USERNAME)).select("parent_id").tail(1)[0][0]

incReadDf = spark.read\
    .format("iceberg")\
    .option("start-snapshot-id", parent_id)\
    .option("end-snapshot-id", snapshot_id)\
    .load("{0}.transactions_{1}".format(DBNAME, USERNAME))

# ----------------------------
# Data prep and model training
# ----------------------------
df = incReadDf.toPandas()

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("fraud_trx", axis=1),
    df["fraud_trx"],
    test_size=0.3
)

model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_proba = model.predict_proba(X_test)[:, 1]

# ----------------------------
# Initialize Dash app
# ----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive Model ROI Dashboard"),
    html.H4("Net Revenue = TP Revenue + TN Revenue - FP Penalty - FN Opportunity Cost",
            style={"font-style": "italic", "color": "gray"}),

    # Threshold slider
    html.Label("Decision Threshold:"),
    dcc.Slider(
        id="threshold-slider",
        min=0.0, max=1.0, step=0.01, value=0.5,
        marks={0: "0.0", 0.25: "0.25", 0.5: "0.5", 0.75: "0.75", 1: "1.0"},
        tooltip={"placement": "bottom", "always_visible": True}
    ),

    # Revenue for class 1
    html.Div([
        html.Label("Financial Revenue (Actual Target=1):"),
        dcc.Input(id="revenue-class-1", type="number", value=100, step=10)
    ], style={"margin-top": "20px"}),

    # Revenue for class 0
    html.Div([
        html.Label("Financial Revenue (Actual Target=0):"),
        dcc.Input(id="revenue-class-0", type="number", value=10, step=10)
    ], style={"margin-top": "10px"}),

    # Penalty for false positives
    html.Div([
        html.Label("Penalty per False Positive:"),
        dcc.Input(id="penalty-fp", type="number", value=50, step=10)
    ], style={"margin-top": "10px"}),

    html.Div(id="accuracy-text", style={"margin-top": "20px", "font-size": "18px"}),

    html.Div(id="breakdown-text", style={"margin-top": "10px", "font-size": "16px", "color": "darkblue"}),

    dcc.Graph(id="confusion-matrix-heatmap")
])

# ----------------------------
# Callbacks
# ----------------------------
@app.callback(
    [Output("accuracy-text", "children"),
     Output("breakdown-text", "children"),
     Output("confusion-matrix-heatmap", "figure")],
    [Input("threshold-slider", "value"),
     Input("revenue-class-1", "value"),
     Input("revenue-class-0", "value"),
     Input("penalty-fp", "value")]
)
def update_outputs(threshold, revenue_1, revenue_0, penalty_fp):
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Revenues
    revenue_tp = tp * revenue_1
    revenue_tn = tn * revenue_0

    # Penalty for FP
    total_fp_penalty = fp * penalty_fp

    # Opportunity cost for FN
    total_fn_opp_cost = fn * revenue_1

    # Net Revenue
    net_revenue = revenue_tp + revenue_tn - total_fp_penalty - total_fn_opp_cost

    # Create confusion matrix dataframe
    cm_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"]
    )

    # Heatmap
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix"
    )

    acc_text = (
        f"Accuracy at threshold {threshold:.2f}: {accuracy:.3f} "
        f"| TP={tp}, TN={tn}, FP={fp}, FN={fn} "
        f"| Net Revenue = ${net_revenue:,.2f}"
    )

    breakdown_text = (
        f"TP Revenue = ${revenue_tp:,.2f}  |  "
        f"TN Revenue = ${revenue_tn:,.2f}  |  "
        f"FP Penalty = ${total_fp_penalty:,.2f}  |  "
        f"FN Opportunity Cost = ${total_fn_opp_cost:,.2f}  |  "
        f"Net Revenue = ${net_revenue:,.2f}"
    )

    return acc_text, breakdown_text, fig

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=int(os.environ["CDSW_APP_PORT"]))
