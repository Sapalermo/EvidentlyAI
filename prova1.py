import pandas as pd
import datetime

from sklearn import ensemble
from evidently.metrics import DatasetMissingValuesMetric, ColumnDriftMetric, DatasetDriftMetric, ColumnSummaryMetric
from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.workspace import WorkspaceBase, Workspace

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

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

#from sklearn.model_selection import train_test_split

#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)

WORKSPACE = "workspace"
YOUR_PROJECT_NAME = "email spam"
YOUR_PROJECT_DESCRIPTION = "prova"

target = 'target'
numerical_features = ['num_char']
categorical_features = ['target']
reference = df.loc[0:4000]
current = df.loc[4000:]

#regressor = ensemble.RandomForestRegressor(random_state = 42, n_estimators = 50)
#regressor.fit(reference[numerical_features + categorical_features], reference[target])
#ref_prediction = regressor.predict(reference[numerical_features + categorical_features])
#current_prediction = regressor.predict(current[numerical_features + categorical_features])
#reference['target'] = ref_prediction
#current['target'] = current_prediction

column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.numerical_features = numerical_features
column_mapping.categorical_features = categorical_features

def create_project(workspace: WorkspaceBase):
    project = workspace.create_project(YOUR_PROJECT_NAME)
    project.description = YOUR_PROJECT_DESCRIPTION
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
    return project

def create_report(i: int):
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="target", stattest_threshold=0.3),
            ColumnSummaryMetric(column_name="target"),
            ColumnDriftMetric(column_name="num_char", stattest_threshold=0.3),
            ColumnSummaryMetric(column_name="num_char"),
        ],
        timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    data_drift_report.run(reference_data=reference,
           current_data=current.loc[0:5575],
           column_mapping=column_mapping)
    return data_drift_report

def create_test_suite(i: int):
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    data_drift_test_suite.run(reference_data=reference,
           current_data=current.loc[0:5575],
           column_mapping=column_mapping)
    return data_drift_test_suite

def create_demo_project(workspace: str):
    ws = Workspace.create(workspace)
    project = create_project(ws)
    for i in range(0, 5):
        report = create_report(i=i)
        ws.add_report(project.id, report)
        suite = create_test_suite(i=i)
        ws.add_report(project.id, suite)

if __name__ == "__main__":
    create_demo_project(WORKSPACE)
