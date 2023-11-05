
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch.optim as optim

from PIL import Image
from model.cnn import ConvNet

from model.dataloader import load_data
from model.train import train

def song_to_spectrogram(song_path,length_sec=30,save_image=True,save_name="spectrogram.png"):


    y, sr = librosa.load(song_path)
    y=y[:sr*length_sec  ]

    S = librosa.feature.melspectrogram(y=y, sr=sr,hop_length=512)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)

    # Desired width in pixels
    desired_width = 336

    # Calculate the DPI needed to achieve the desired width
    dpi = int(desired_width / plt.figure(figsize=(desired_width / 80, 4)).get_figwidth())
    plt.clf()

    width=desired_width / dpi
    fig=plt.figure(figsize=(width, width/3*2))
    librosa.display.specshow(S_DB, sr=sr,hop_length=512,
                             x_axis='time', y_axis='mel')
    plt.gca().set_axis_off()
    #plt.colorbar()
    #plt.savefig("spectrogram.png", bbox_inches='tight', pad_inches=0, transparent=False)
    #plt.title("Mel spectrogram", fontsize=20)
    #plt.show()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    image = Image.fromarray(image_array)

    # Define the new size
    new_size = (432, 288)  # Change this to your desired dimensions

    # Resize the image
    resized_image = image.resize(new_size, Image.LANCZOS)
    resized_image_data = np.array(resized_image)
    #resized_image.show()  # Opens the image using the default viewer
    if save_image:
        resized_image.save(save_name)
        print("spectrogram saved at "+save_name)

    return resized_image_data