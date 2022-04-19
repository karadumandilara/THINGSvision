from thingsvision.custom_models.custom import Custom
import tensorflow as tf

class SimCLR(Custom):
    def __init__(self, device, backend) -> None:
        super().__init__(device, backend)

    def create_model(self):
        if self.backend == 'tf':
            saved_model_path = 'gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/r50_1x_sk0/saved_model/'
            model = tf.keras.models.load_model(saved_model_path)
            #model = tf.saved_model.load(saved_model_path)
            return model