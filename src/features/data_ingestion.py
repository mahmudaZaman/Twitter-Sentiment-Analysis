from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_uri: str = "/Users/shuchi/Documents/work/personal/Twitter-Sentiment-Analysis/dataset/train.csv"
    print("train_data_uri", train_data_uri)
    test_data_uri: str = "/Users/shuchi/Documents/work/personal/Twitter-Sentiment-Analysis/dataset/test.csv"
    print("test_data_uri", test_data_uri)

    # train_data_uri: str = f"s3://{app_config.storage.bucket_name}/{app_config.storage.files.train_data}"
    # print("train_data_uri", train_data_uri)
    # test_data_uri: str = f"s3://{app_config.storage.bucket_name}/{app_config.storage.files.test_data}"
    # print("test_data_uri", test_data_uri)


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        train_df = pd.read_csv(self.ingestion_config.train_data_uri)
        test_df = pd.read_csv(self.ingestion_config.test_data_uri)

        # shuffle train data
        train_df = train_df.sample(frac=1, random_state=42)
        print(train_df.head())

        # train test split
        X = train_df["text"]
        y = train_df["target"]
        train_sentences, test_sentences, train_label, test_label = train_test_split(X, y,
                                                                                    test_size=0.1,
                                                                                    random_state=42)
        print(train_sentences[:5], train_label[:5])
        print(type(train_sentences), type(train_label))
        return (
            train_sentences, test_sentences, train_label, test_label
        )
