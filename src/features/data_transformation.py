from sklearn.model_selection import train_test_split

class DataTransformation:
    @staticmethod
    def initiate_data_transformation(train_sentences, test_sentences, train_label, test_label):  # Todo verify
        # convert to numpy array
        train_sentences, test_sentences, train_label, test_label = train_sentences.to_numpy(), test_sentences.to_numpy(), train_label.to_numpy(), test_label.to_numpy()
        print(train_sentences[:5], train_label[:5])
        print(type(train_sentences), type(train_label))

        return (
            train_sentences, test_sentences, train_label, test_label
        )
