from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import numpy as np
import json
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from pyod.models.iforest import IForest
import joblib
import torch
from torch import nn
from torch import autograd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def extract_data(**kwargs):
    ti = kwargs['ti']
    current_file_path = os.path.abspath(__file__)

    csv_file_path = os.path.join(os.path.dirname(current_file_path), 'merged_transaction_data.csv')
    banned_csv_file_path = os.path.join(os.path.dirname(current_file_path), 'banned_districts.csv')
    banned_districts_df = pd.read_csv(banned_csv_file_path)
    banned_district_ids = banned_districts_df['district_id'].tolist()

    df = pd.read_csv(csv_file_path)
    df_filtered_transactions = df[~df['district_id'].isin(banned_district_ids)]
    
    # Convert DataFrame to JSON string
    df_filtered_transactions_json = df_filtered_transactions.to_json(orient='records')
    
    return df_filtered_transactions_json


def transform_data(**kwargs):
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='extract_data')
    df = pd.read_json(df_json)
    df = df[['trans_date', 'account_id', 'trans_type', 'amount']]
    df['trans_date'] = pd.to_datetime(df['trans_date'])
    rename = {'C': 'CREDIT', 'D': 'WITHDRAWAL', 'P': 'NOT SURE'}
    df['trans_type'] = df['trans_type'].replace(rename)
    print(df.head())
    df_withdrawals = df.query('trans_type == "WITHDRAWAL"').sort_values(by=['account_id', 'trans_date']).set_index('trans_date')
    df_withdrawals['amount'] = df_withdrawals['amount'].abs()
    df_withdrawals['amount'] = df_withdrawals['amount']*10
    print(df_withdrawals.head())
    df_withdrawals['total_withdrawals_5d'] = df_withdrawals.groupby('account_id')['amount'].transform(lambda s: s.rolling(timedelta(days=5)).sum())

    df_withdrawals['num_withdrawals_5d'] = df_withdrawals.groupby('account_id')['amount'].transform(lambda s: s.rolling(timedelta(days=5)).count())

    print(df_withdrawals.head())
    account_data = df_withdrawals.groupby('account_id').agg({
        'num_withdrawals_5d': 'sum',
        'total_withdrawals_5d': 'sum', # You can add more aggregated features if needed
    }).reset_index()


    # Assuming df_withdrawals is your DataFrame containing the data
    ratio = 0.55

    # Split the data into training and test sets
    df_withdrawals_train, df_withdrawals_test = train_test_split(df_withdrawals, test_size=1-ratio, random_state=42)

    # Reset index
    df_withdrawals_train = df_withdrawals_train.reset_index(drop=True)
    df_withdrawals_test = df_withdrawals_test.reset_index(drop=True)
  # Reset index here
    scaler = StandardScaler()
    X = account_data[['num_withdrawals_5d', 'total_withdrawals_5d']]
    X_scaled = scaler.fit_transform(X)
    account_data_json = account_data.to_json()
    df_withdrawals_train_json = df_withdrawals_train.to_json(orient='records')
    df_withdrawals_test_json = df_withdrawals_test.to_json(orient='records')
    ti.xcom_push(key='account_data', value=account_data_json)
    ti.xcom_push(key='df_withdrawals_train_json', value=df_withdrawals_train_json)
    ti.xcom_push(key='df_withdrawals_test_json', value=df_withdrawals_test_json)
    X_scaled_list = X_scaled.tolist()
    X_scaled_json = json.dumps(X_scaled_list)
    return X_scaled_json

"""def transform_sap_data(**kwargs):
    ti = kwargs['ti']
    df_json = ti.xcom_pull(task_ids='extract_sap_data')
    df = pd.read_json(df_json)
    label = df.pop('label')
    # select categorical attributes to be "one-hot" encoded
    categorical_attr_names = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT','WAERS', 'BUKRS']

    # encode categorical attributes into a binary one-hot encoded representation 
    ori_dataset_categ_transformed = pd.get_dummies(df[categorical_attr_names])
    # select "DMBTR" vs. "WRBTR" attribute
    numeric_attr_names = ['DMBTR', 'WRBTR']

    # add a small epsilon to eliminate zero values from data for log scaling
    numeric_attr = df[numeric_attr_names] + 1e-7
    numeric_attr = numeric_attr.apply(np.log)

    # normalize all numeric attributes to the range [0,1]
    ori_dataset_numeric_attr = (numeric_attr - numeric_attr.min()) / (numeric_attr.max() - numeric_attr.min())
    ori_subset_transformed = pd.concat([ori_dataset_categ_transformed, ori_dataset_numeric_attr], axis = 1)
    ori_subset_transformed_numeric = ori_subset_transformed.astype(int)

class encoder(nn.Module):

    def __init__(self):

        super(encoder, self).__init__()

        # specify layer 1 - in 618, out 3
        self.encoder_L1 = nn.Linear(in_features=ori_subset_transformed.shape[1], out_features=3, bias=True) # add linearity 
        nn.init.xavier_uniform_(self.encoder_L1.weight) # init weights according to [9]
        self.encoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) # add non-linearity according to [10]
        
    def forward(self, x):

        # define forward pass through the network
        x = self.encoder_R1(self.encoder_L1(x)) # don't apply dropout to the AE bottleneck

        return x
    
# implementation of the shallow decoder network 
# containing only a single layer
class decoder(nn.Module):

    def __init__(self):

        super(decoder, self).__init__()

        # specify layer 1 - in 3, out 618
        self.decoder_L1 = nn.Linear(in_features=3, out_features=ori_subset_transformed.shape[1], bias=True) # add linearity 
        nn.init.xavier_uniform_(self.decoder_L1.weight)  # init weights according to [9]
        self.decoder_R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) # add non-linearity according to [10]

    def forward(self, x):

        # define forward pass through the network
        x = self.decoder_R1(self.decoder_L1(x)) # don't apply dropout to the AE output
        
        return x

def encoder_decoder_train_init(**kwargs):
    encoder_train = encoder()
    decoder_train = decoder()
    now = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
    print('[LOG {}] decoder architecture:\n\n{}\n'.format(now, decoder_train))
    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    learning_rate = 1e-3
    encoder_optimizer = torch.optim.Adam(encoder_train.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder_train.parameters(), lr=learning_rate)
    num_epochs = 5
    mini_batch_size = 128"""


def clustering(**kwargs):
    ti = kwargs['ti']
    X_scaled_json = ti.xcom_pull(task_ids='transform_data')
    X_scaled = pd.read_json(X_scaled_json)
    account_data = ti.xcom_pull(task_ids='transform_data', key='account_data')
    account_data = pd.read_json(account_data)
    df_withdrawals_train_json = ti.xcom_pull(task_ids='transform_data', key='df_withdrawals_train_json')
    df_withdrawals = pd.read_json(df_withdrawals_train_json)
    num_clusters = 10  # You can adjust the number of clusters
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    account_data['cluster'] = kmeans.fit_predict(X_scaled)
    print("Clustered data:")
    print(account_data.head())
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    cluster_dir = os.path.join(data_dir, 'cluster_data')
    os.makedirs(cluster_dir, exist_ok=True)

    for cluster_id in range(num_clusters):
        cluster_accounts = account_data[account_data['cluster'] == cluster_id]['account_id']
        cluster_transactions = df_withdrawals[df_withdrawals['account_id'].isin(cluster_accounts)]
        
        cluster_filename = os.path.join(cluster_dir, f'cluster_{cluster_id}_withdrawals.csv')
        cluster_transactions.to_csv(cluster_filename, index=True)
    cluster_dir_path = os.path.abspath(cluster_dir)
    print(f"Cluster data folder path: {cluster_dir_path}")
    first_cluster_filename = os.path.join(cluster_dir, f'cluster_0_withdrawals.csv')
    if os.path.exists(first_cluster_filename):
        first_cluster_data = pd.read_csv(first_cluster_filename)
        print(f"Head of the first cluster:\n{first_cluster_data.head()}")
    else:
        print("Cluster data not found.")

def isolation_forest(**kwargs):
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    cluster_dir = os.path.join(data_dir, 'cluster_data')
    result_dir = os.path.join(data_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')) 
    model_dir = os.path.join(models_dir, 'saved_models')
    os.makedirs(model_dir, exist_ok=True)
    anomaly_proportion = 0.001

    for filename in os.listdir(cluster_dir):
        if filename.endswith('.csv'):
            cluster_data = pd.read_csv(os.path.join(cluster_dir, filename))
        
            X = cluster_data[['num_withdrawals_5d', 'total_withdrawals_5d']]
            clf = IForest(contamination=anomaly_proportion)
            clf.fit(X)
            
            # anomaly labels and scores
            cluster_data['y_pred'] = clf.labels_
            cluster_data['y_scores'] = clf.decision_scores_
            
            # Saving the results
            result_filename = os.path.join(result_dir, f'anomaly_detection_{filename}')
            cluster_data.to_csv(result_filename, index=False)
            
            # Extract cluster number from filename
            cluster_number = filename.split('_')[-2]
            
            model_filename = os.path.join(model_dir, f'iforest_model_{cluster_number}.joblib')
            joblib.dump(clf, model_filename)



def test_withdrawals_test(**kwargs):
    ti = kwargs['ti']
    df_withdrawals_test_json = ti.xcom_pull(task_ids='transform_data', key='df_withdrawals_test_json')
    df_withdrawals_test = pd.read_json(df_withdrawals_test_json).head(50)  # Selecting the first 50 rows
    
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    model_dir = os.path.join(data_dir, 'saved_models')
    
    results = []  # List to store results
    
    for index, row in df_withdrawals_test.iterrows():
        test_data_point = pd.DataFrame({
            'account_id': [row['account_id']],
            'trans_type': ['WITHDRAWAL'],
            'amount': [row['amount']],
            'total_withdrawals_5d': [row['total_withdrawals_5d']],
            'num_withdrawals_5d': [row['num_withdrawals_5d']]
        })
        
        cluster_id_col = 'cluster'
        cluster_id = 0  
        for filename in os.listdir(model_dir):
            if filename.endswith('.joblib'):
                try:
                    cluster_id_from_filename = int(filename.split('_')[3])
                except IndexError:
                    continue

                model = joblib.load(os.path.join(model_dir, filename))

                # Check if the test account ID belongs to this cluster
                if test_data_point['account_id'].iloc[0] in model.labels_:
                    cluster_id = cluster_id_from_filename
                    break

        model_filename = os.path.join(model_dir, f'iforest_model_{cluster_id}.joblib')
        model = joblib.load(model_filename)
        X_test = test_data_point[['num_withdrawals_5d', 'total_withdrawals_5d']]

        y_pred = model.predict(X_test)
        
        # Append result to list
        results.append({
            'account_id': row['account_id'],
            'trans_type': row['trans_type'],
            'amount': row['amount'],
            'total_withdrawals_5d': row['total_withdrawals_5d'],
            'num_withdrawals_5d': row['num_withdrawals_5d'],
            'anomaly_label': int(y_pred == 1)  # 0 if not anomaly, 1 if anomaly
        })
    
    # Convert results to DataFrame
    df_results = pd.DataFrame(results)
    
    # Save results to CSV file
    results_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'results', 'withdrawals_test_results.csv'))
    df_results.to_csv(results_csv_path, index=False)

    print(f"Results saved to: {results_csv_path}")



default_args = {
    'owner': 'your_name',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG("anomaly_detection_dag", default_args=default_args, description='ML pipeline for anomaly detection',
         schedule_interval='@daily', catchup=False) as dag:
    extract_data_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        provide_context=True,
    )
    transform_data_task = PythonOperator(
        task_id='transform_data',
        python_callable=transform_data,
        provide_context=True,
    )
    clustering_task = PythonOperator(
        task_id='clustering',
        python_callable=clustering,
        provide_context=True,
    )
    isolation_forest_task = PythonOperator(
        task_id='isolation_forest',
        python_callable=isolation_forest,
        provide_context=True,
    )
    test_withdrawals_test_task = PythonOperator(
        task_id='test_withdrawals_test',
        python_callable=test_withdrawals_test,
        provide_context=True,
        dag=dag
    )


    extract_data_task >> transform_data_task >> clustering_task >> isolation_forest_task >> test_withdrawals_test_task