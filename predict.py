import gradio as gr
import keras.utils as image
import numpy as np
import tensorflow as tf

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('/Users/alpaca/Downloads/kagglecatsanddogs_5340/epochs50.h5')

# Show the model architecture
model.summary()


def predictFun(image_file):
    # test_image = image.load_img(image_file, target_size=(200, 200))
    test_image = image.img_to_array(image_file)
    test_image = np.expand_dims(test_image, axis=0)

    # Result array
    result = model.predict(test_image)
    result = {'Dog': float(result[0][0]), 'Cat': float(1 - result[0][0])}
    # if result >= 0.5:
    #     print("Dog")
    # else:
    #     print("Cat")
    # print(result)
    return result


inputs = gr.inputs.Image()
outputs = gr.outputs.Label(num_top_classes=2)

gr.Interface(fn=predictFun,
             inputs=gr.Image(shape=(200, 200)),
             outputs=gr.Label(num_top_classes=2),
             ).launch(share=True)
