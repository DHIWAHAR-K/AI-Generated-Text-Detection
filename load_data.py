import pandas as pd

def load_datasets():
    test = pd.read_csv('/data_2/test_essays.csv')
    sub = pd.read_csv('/data_4/sample_submission.csv')
    org_train = pd.read_csv('/data_2/train_essays.csv')
    train1 = pd.read_csv("/data_1/train_v2_drcat_02.csv", sep=',')
    train2 = pd.read_csv('/data_1/train_drcat_04.csv')

    org_train = org_train.rename(columns={'generated': 'label'})
    excluded_prompts = ['Distance learning','Grades for extracurricular activities','Summer projects']
    train1 = train1[~train1['prompt_name'].isin(excluded_prompts)]

    train = pd.concat([org_train, train1, train2])
    train = train.drop_duplicates(subset=['text']).reset_index(drop=True)
    return train, test, sub