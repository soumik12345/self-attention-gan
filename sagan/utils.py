import matplotlib.pyplot as plt
import tensorflow as tf

## Save the images
def save_plot(examples, epoch, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        plt.subplot(n, n, i + 1)
        plt.axis("off")
        plt.imshow(examples[i])
    filename = f"/samples/demo-{epoch+1}.png"
    plt.savefig(filename)
    plt.close()

def w_loss(y_true, y_pred):
    return tf.reduce_mean(y_true * y_pred)