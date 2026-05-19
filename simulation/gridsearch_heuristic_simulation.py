
import subprocess
import sys
import src.const as srconst
from scipy.stats import wasserstein_distance
import configparser
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def main():
    # lists
    lists_interaction_retweet_weights =[0.1, 0.25, 0.5, 0.75, 0.9]
    lists_interaction_reply_weights_pos = [0.1, 0.25, 0.5, 0.75, 0.9]
    lists_interaction_reply_weights_neutral = [0, 0.1]
    lists_stubbornness_retweet_weights = [0.1, 0.25, 0.5, 0.75, 0.9]
    lists_stubbornness_reply_weights = [0.1, 0.25, 0.5, 0.75, 0.9]

    interaction_reply_weights_neg = 0

    TEMP_CONFIG_PATH = 'configs/temp_fj_config.ini'
    MODEL = 'Friedkin-Johnsen'
    h_timestep=201


    config = configparser.ConfigParser()

    config['DATA_PATH'] = {
        'text_data': 'data/text_data_sentiment_topics.parquet',
        'author_data': 'data/author_data.parquet',
        'interaction_data': 'data/interaction_data.parquet',
    }

    
    # loop
    for topic in ['all','0','1','2','3','4']:
        gridsearch_df = []
        print('topic:',topic)
        config['SIMULATION'] = {
            'init_days': '7',
            'topic_num': topic,
        }
        log_file = f'logs/gridsearch/log_{topic}.txt'
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        pbar = tqdm(total=len(lists_interaction_retweet_weights) * len(lists_interaction_reply_weights_pos) * len(lists_interaction_reply_weights_neutral) * len(lists_stubbornness_retweet_weights) * len(lists_stubbornness_reply_weights), desc="Tracking Grid Search")
        for i in lists_interaction_retweet_weights:
            for j in lists_interaction_reply_weights_pos:
                for k in lists_interaction_reply_weights_neutral:
                        for m in lists_stubbornness_retweet_weights:
                            for n in lists_stubbornness_reply_weights:
                                pbar.update(1)
                                package = {
                                    'model': MODEL,
                                    'tolerance': '1e-6',
                                    'interaction_retweet_weights': str(i),
                                    'interaction_reply_weights_pos': str(j),
                                    'interaction_reply_weights_neutral': str(k),
                                    'interaction_reply_weights_neg': str(interaction_reply_weights_neg),
                                    'stubbornness_retweet_weights': str(m),
                                    'stubbornness_reply_weights': str(n),
                                }

                                config['MODEL'] = package
                                with open(TEMP_CONFIG_PATH, 'w') as f:
                                    config.write(f)

                                venv_path = "venv/bin/activate"
                                config_file = TEMP_CONFIG_PATH
                                command = f"source {venv_path}; CONFIG=\"{config_file}\"; python -m simulation.heuristic_simulation -c \"$CONFIG\""
                                subprocess.run(["bash", "-l", "-c", command], stdout=open(log_file, "a"), stderr=subprocess.STDOUT)
                                
                                df_model = pd.read_csv(f'outputs/heuristic_{MODEL}/{topic}/opinion_shift_step_{h_timestep}.csv',sep=';')
                                df_model.rename(columns={'datetime':'time_step'},inplace=True)
                                df_model.drop(columns=['topic'],inplace=True)
                                df_model['time_step'] = df_model['time_step'].astype('datetime64[ns]')
                                df_model['date'] = df_model['time_step'].dt.date
                                df_model2 = df_model.groupby(['date','agent'],as_index=False)['opinion_weight'].mean()


                                ground_df = pd.read_csv(f'data/ground_truth/baseline_{topic}.csv',sep=';')
                                ground_df = ground_df.rename(columns={'datetime':'date','sentiment_label':"opinion_weight"}).copy()
                                ground_df = ground_df.sort_values(['date','agent'],ignore_index=True).copy()
                                
                                linew = []
                                for date in range(6,9):
                                    temp_date = '2024-02-'+str(date+1).zfill(2)
                                    A = ground_df.loc[pd.to_datetime(ground_df['date'])==temp_date,'opinion_weight']
                                    B = df_model2.loc[pd.to_datetime(df_model2['date'])==temp_date,'opinion_weight']
                                    w = wasserstein_distance(A, B)
                                    linew.append(round(w,5))
                                mae0 = np.average(np.abs(linew))
                                
                                package['mae0'] = mae0
                                package['topic'] = topic
                                gridsearch_df.append(package)

        gridsearch_df = pd.DataFrame(gridsearch_df)
        gridsearch_df.to_csv(f'outputs/gridsearch/gridsearch_{MODEL}_{topic}.csv',sep=';',index=False)

if __name__ == "__main__":
    main()