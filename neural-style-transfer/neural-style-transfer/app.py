import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import gradio as gr
import cv2


IMAGE_SIZE = (256, 256)


style_transfer_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def load_image(image):
    
    image = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    
    image = image.astype(np.float32)[np.newaxis, ...] / 255.
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image

def apply_sharpness(image, intensity):
    kernel = np.array([[0, -intensity, 0],
                       [-intensity, 1 + 4 * intensity, -intensity],
                       [0, -intensity, 0]])
    sharp_image = cv2.filter2D(image, -1, kernel)
    return np.clip(sharp_image, 0, 255)

def interpolate_images(baseline, target, alpha):
    
    return baseline + alpha * (target - baseline)

def style_transfer(content_image, style_image, style_density, content_sharpness):
    #
    content_image = load_image(content_image)
    style_image = load_image(style_image)

    
    content_image_sharp = apply_sharpness(content_image[0], intensity=content_sharpness)
    content_image_sharp = content_image_sharp[np.newaxis, ...]

    
    stylized_image = style_transfer_model(tf.constant(content_image_sharp), tf.constant(style_image))[0]

    
    stylized_image = interpolate_images(
        baseline=content_image[0],
        target=stylized_image.numpy(),
        alpha=style_density
    )

    
    stylized_image = np.array(stylized_image * 255, np.uint8)

    
    stylized_image = np.squeeze(stylized_image)
    return stylized_image

iface = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(label="Content Image"),  
        gr.Image(label="Style Image"),    
        gr.Slider(minimum=0, maximum=1, value=0.5, label="Adjust Style Density"),
        gr.Slider(minimum=0, maximum=1, value=0.5, label="Content Sharpness")
    ],
    outputs=gr.Image(label="Stylized Image")
)

iface.launch()
