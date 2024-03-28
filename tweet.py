import glob
import os.path
import csv
from evidently import ColumnMapping
import random
import pandas as pd
from evidently.tests import *
from requests.exceptions import RequestException
from evidently.metrics import *
from evidently.collector.client import CollectorClient
from evidently.collector.config import CollectorConfig
from evidently.collector.config import IntervalTrigger
from evidently.collector.config import ReportConfig
from evidently.metric_preset import *
from evidently.test_preset import *
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import *
from evidently.ui.workspace import Workspace
from evidently.descriptors import *
from evidently.calculations.stattests import *

df1=pd.read_csv('./training_set_sentipolc16.csv', encoding='iso-8859-1')
df2=pd.read_csv('./test_set_sentipolc16_gold2000.csv', on_bad_lines='skip')

# inserimento nomi colonne
df2.columns=["idtwitter","subj","opos","oneg","iro","lpos","lneg","top","text"]

# Concatena i due dataset
merged_df = pd.concat([df1,df2])

# Salva il dataset concatenato in un nuovo file CSV
merged_df.to_csv('merged_dataset.csv', index=False)

# Percorso della directory contenente i file CSV
main_directory = './dataset/40wita dataset - 1st covid year'

# Inizializza un DataFrame vuoto per contenere i dati concatenati
df_concatenato = pd.DataFrame()

# Ottieni i file CSV nella directory
csv_files = glob.glob(os.path.join(main_directory, '*.csv'))

# Definisco la dimensione del campione
sample_size = 10

# Itera su tutti i file CSV
for csv_file in csv_files:
    # Legge il file CSV
    data = pd.read_csv(csv_file, quoting=csv.QUOTE_NONE, on_bad_lines='skip',  lineterminator='\n')
    
    # Controlla se il file ha almeno 10 righe
    if len(data) >= sample_size:
        # Effettua il random sampling di 10 righe
        sampled_rows = random.sample(range(len(data)), sample_size)
        sampled_data = data.iloc[sampled_rows]
    else:
        # Se il file ha meno di 10 righe, utilizza tutte le righe
        sampled_data = data
    
    # Aggiungi i dati campionati al DataFrame concatenato
    df_concatenato = pd.concat([df_concatenato, sampled_data], ignore_index=True)

# Salva il DataFrame concatenato in un nuovo file CSV
df_concatenato.to_csv('./concatenated_dataset.csv', index=False)

COLLECTOR_ID = "default"
COLLECTOR_TEST_ID = "default_test"

PROJECT_NAME = "tweet covid"

WORKSPACE_PATH = "workspace"
LINK_URL = "http://localhost:8000/"

"""import nltk
nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')"""

client = CollectorClient("http://localhost:8001")

target = 'text'
text_features=['text']

column_mapping = ColumnMapping()

column_mapping.target = target
column_mapping.text_features = text_features

def setup_report():
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(columns=["text"]),
        ]
    )
    data_drift_report.run(reference_data=merged_df,
           current_data=df_concatenato,
           column_mapping=column_mapping)

    return ReportConfig.from_report(data_drift_report)


def setup_test_suite():
    data_drift_test_suite = TestSuite(
        tests=[TestShareOfDriftedColumns(columns=["text"]),
            ],
    )
    data_drift_test_suite.run(reference_data=merged_df,
           current_data=df_concatenato,
           column_mapping=column_mapping)

    return ReportConfig.from_test_suite(data_drift_test_suite)


def setup_workspace():
    ws = Workspace.create(WORKSPACE_PATH)
    project = ws.create_project(PROJECT_NAME)
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Confronto tra tweet del modello BehaViz_neuraly e tweet estrapolati dal periodo del Covid-19",
        )
    )
    project.save()

def setup_config():
    ws = Workspace.create(WORKSPACE_PATH)
    project = ws.search_project(PROJECT_NAME)[0]
    conf = CollectorConfig(
        trigger=IntervalTrigger(interval=3), report_config=setup_report(), project_id=str(project.id)
    )
    client.create_collector(id=COLLECTOR_ID, collector=conf)
    test_conf = CollectorConfig(
        trigger=IntervalTrigger(interval=3), report_config=setup_test_suite(), project_id=str(project.id)
    )
    client.create_collector(id=COLLECTOR_TEST_ID, collector=test_conf)


    client.set_reference(id=COLLECTOR_ID, reference=merged_df)
    client.set_reference(id=COLLECTOR_TEST_ID, reference=merged_df)


def send_data():

    client.send_data(COLLECTOR_ID, df_concatenato)
    client.send_data(COLLECTOR_TEST_ID, df_concatenato)
    
    print("sent")


def start_sending_data():
    print("Start data")
    while True:
        try:
            send_data()
            break
        except RequestException as e:
            print(f"collector service not available: {e.__class__.__name__}")

    #creare while che si ferma dopo la prima iterazione
        
        
        
        

def main():
        if not os.path.exists(WORKSPACE_PATH) or len(Workspace.create(WORKSPACE_PATH).search_project(PROJECT_NAME)) == 0:
            setup_workspace()

        setup_config()

        start_sending_data()
    


if __name__ == "__main__":
    main()



