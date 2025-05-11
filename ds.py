import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(r"D:\Projects\ATHESIS\Assistant\models\fire_classification_mobilenet_v3")

# Activează suportul pentru op-uri TF avansate (inclusiv LSTM)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,       # op-uri normale TFLite
    tf.lite.OpsSet.SELECT_TF_OPS          # op-uri din TF care nu sunt suportate nativ de TFLite
]

# Dezactivează experimental lowering pentru TensorList (esential pentru LSTM)
converter._experimental_lower_tensor_list_ops = False

# (opțional) activează optimizarea
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convertește
tflite_model = converter.convert()

# Salvează modelul
with open(r"D:\Projects\ATHESIS\Assistant\models\fire_classification_mobilenet_v3.tflite", "wb") as f:
    f.write(tflite_model)
