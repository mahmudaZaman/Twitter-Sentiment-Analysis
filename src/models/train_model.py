import tensorflow as tf
import tensorflow_hub as hub
from src.models.callback import checkpoint_path, create_callbacks
from src.features.data_ingestion import DataIngestion
from src.features.data_transformation import DataTransformation


class ModelTrainer:
    def initiate_model_trainer(self, train_sentences, test_sentences, train_label, test_label):
        sentence_encoder_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",
                                                input_shape=[],
                                                dtype=tf.string,
                                                trainable=False,
                                                name="USE")

        # functional
        inputs = tf.keras.layers.Input(shape=[], dtype=tf.string)
        pretrained_embedding = sentence_encoder_layer(inputs)  # tokenize text and create embedding
        x = tf.keras.layers.Dense(128, activation="relu")(
            pretrained_embedding)  # add a fully connected layer on top of the embedding
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)  # create the output layer
        use_model = tf.keras.Model(inputs=inputs, outputs=outputs)

        use_model.compile(loss="binary_crossentropy",
                          optimizer=tf.keras.optimizers.Adam(),
                          metrics=["accuracy"])

        callbacks = create_callbacks()

        use_model.fit(train_sentences,
                      train_label,
                      epochs=5,
                      validation_data=(test_sentences, test_label),
                      callbacks=callbacks)

        use_model.evaluate(test_sentences, test_label)
        use_model_probs = use_model.predict(test_sentences)
        print(use_model_probs[:10])

        use_model_pred = tf.squeeze(tf.round(use_model_probs))
        print(use_model_pred[:20])

        use_model.save("out/tweet_model.h5")
        use_model.load_weights(checkpoint_path)
        loaded_weights_model_results = use_model.evaluate(test_sentences, test_label)
        print("loaded_weights_model_results: ", loaded_weights_model_results)


def run_train_pipeline():
    obj = DataIngestion()
    train_sentences, test_sentences, train_label, test_label = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_sentences, test_sentences, train_label, test_label = data_transformation.initiate_data_transformation(train_sentences, test_sentences, train_label, test_label)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_sentences, test_sentences, train_label, test_label)
    print("model training completed")


# if __name__ == '__main__':
#     run_train_pipeline()