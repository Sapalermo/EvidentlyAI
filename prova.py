import datetime
import os.path
import time
from evidently import ColumnMapping

import pandas as pd
from requests.exceptions import RequestException
from evidently.test_preset import DataDriftTestPreset
from evidently.metrics import DatasetMissingValuesMetric, ColumnDriftMetric, DatasetDriftMetric, ColumnSummaryMetric
from evidently.collector.client import CollectorClient
from evidently.collector.config import CollectorConfig
from evidently.collector.config import IntervalTrigger
from evidently.collector.config import ReportConfig
from evidently.collector.config import RowsCountTrigger
from evidently.metrics import ColumnValueRangeMetric
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfOutRangeValues
from evidently.ui.dashboards import DashboardPanelPlot, DashboardPanelCounter, CounterAgg
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.workspace import Workspace

df=pd.read_csv('./email_spam.csv', encoding='iso-8859-1')

# pulizia dei dati
df.drop(columns =['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace= True)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)

# elimina i duplicati
df= df.drop_duplicates(keep='first')

from sklearn.preprocessing import LabelEncoder

encoder= LabelEncoder()
df['target']=encoder.fit_transform(df['target'])

import nltk

nltk.download('punkt')
df['num_char']=df['text'].apply(len)
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))

#preprocessing

from nltk.corpus import stopwords
import string
from nltk.stem.porter  import PorterStemmer

ps = PorterStemmer()
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    
    
    text=y[:]
    y.clear()
    for i in text:
         y.append(ps.stem(i))
       
    
    return " ".join(y)

df['transformed_text']=df['text'].apply(transform_text)


COLLECTOR_ID = "default"
COLLECTOR_TEST_ID = "default_test"

PROJECT_NAME = "prova online spam"

WORKSPACE_PATH = "workspace"

client = CollectorClient("http://localhost:8001")

target = 'target'
numerical_features = ['num_char']
categorical_features = ['target']
reference = df.loc[0:4000]
current = df.loc[4000:]

column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features


""" def get_data():
    # cur = ref = pd.DataFrame([{"values1": 5.0, "values2": 0.0} for _ in range(10)])
    # return cur, ref
    return current, reference """


def setup_report():
    #report = Report(metrics=[ColumnValueRangeMetric("values1", left=5)], tags=["quality"])
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="target", stattest_threshold=0.3),
            ColumnSummaryMetric(column_name="target"),
            ColumnDriftMetric(column_name="num_char", stattest_threshold=0.3),
            ColumnSummaryMetric(column_name="num_char"),
        ],
        #timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    #data_drift_report.run(reference_data=ref, current_data=cur)
    data_drift_report.run(reference_data=reference,
           current_data=current,
           column_mapping=column_mapping)

    return ReportConfig.from_report(data_drift_report)


def setup_test_suite():
    #report = TestSuite(tests=[TestNumberOfOutRangeValues("values1", left=5)], tags=["quality"])
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        #timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    #report.run(reference_data=ref, current_data=cur)
    data_drift_test_suite.run(reference_data=reference,
           current_data=current,
           column_mapping=column_mapping)
    return ReportConfig.from_test_suite(data_drift_test_suite)


def setup_workspace():
    ws = Workspace.create(WORKSPACE_PATH)
    project = ws.create_project(PROJECT_NAME)
    """  project.dashboard.add_panel(
            DashboardPanelPlot(
                title="sample_panel",
                filter=ReportFilter(metadata_values={}, tag_values=["quality"]),
                values=[
                    PanelValue(metric_id="ColumnValueRangeMetric", field_path="current.share_in_range", legend="current"),
                    PanelValue(
                        metric_id="ColumnValueRangeMetric", field_path="reference.share_in_range", legend="reference"
                    ),
                ],
                plot_type=PlotType.LINE,
            )
        )
        project.save() """
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="prova",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Dati mancanti",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetMissingValuesMetric",
                field_path=DatasetMissingValuesMetric.fields.current.number_of_rows,
                legend="count",
            ),
            text="count",
            agg=CounterAgg.SUM,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Target",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "target"},
                    field_path=ColumnDriftMetric.fields,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.HISTOGRAM,
            size=1,
        )
    )
    project.save()

def setup_config():
    ws = Workspace.create(WORKSPACE_PATH)
    project = ws.search_project(PROJECT_NAME)[0]
    # conf = CollectorConfig(trigger=IntervalTrigger(interval=5), report_config=setup_report(), project_id=str(project.id))
    conf = CollectorConfig(
        trigger=IntervalTrigger(interval=5), report_config=setup_report(), project_id=str(project.id)
    )
    client.create_collector(id=COLLECTOR_ID, collector=conf)

    # test_conf = CollectorConfig(trigger=IntervalTrigger(interval=5), report_config=setup_test_suite(), project_id=str(project.id))
    test_conf = CollectorConfig(
        trigger=IntervalTrigger(interval=5), report_config=setup_test_suite(), project_id=str(project.id)
    )
    client.create_collector(id=COLLECTOR_TEST_ID, collector=test_conf)


    client.set_reference(id=COLLECTOR_ID, reference=reference)
    client.set_reference(id=COLLECTOR_TEST_ID, reference=reference)


def send_data():
    #size = 1
    #data = pd.DataFrame([{"target": df['target'], "num_char": df['num_char']} for _ in range(size)])

    client.send_data(COLLECTOR_ID, reference)
    client.send_data(COLLECTOR_TEST_ID, reference)
    client.send_data(COLLECTOR_ID, current)
    client.send_data(COLLECTOR_TEST_ID, current)
    print("sent")


def start_sending_data():
    print("Start data loop")
    while True:
        try:
            send_data()
        except RequestException as e:
            print(f"collector service not available: {e.__class__.__name__}")
        time.sleep(1)

    #creare while che si ferma dopo la prima iterazione
        
        
        
        

def main():
        if not os.path.exists(WORKSPACE_PATH) or len(Workspace.create(WORKSPACE_PATH).search_project(PROJECT_NAME)) == 0:
            setup_workspace()

        setup_config()

        start_sending_data()
    


if __name__ == "__main__":
    main()