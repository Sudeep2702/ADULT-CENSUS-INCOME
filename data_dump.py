import pymongo
import pandas as pd
import json
from census.config import mongo_client
import os 


DATA_FILE_PATH= "/config/workspace/adult.csv"
DATABASE_NAME="adult"
COLLECTION_NAME="census"

if __name__=="__main__":
    df = pd.read_csv("/config/workspace/adult.csv")
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[1:3])
    #insert converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)


