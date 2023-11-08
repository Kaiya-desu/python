"""
A neutral network that identifies Pokémon type
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import argparse
import os
import keras as ks
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator


# ========================   LOAD AND SHOW    =========================
def show_dataset_info(pokedex):
    print(pokedex.info())
    print(pokedex.head())


def show_example_images(image_folder):
    images = os.listdir(image_folder)
    images.remove('.DS_Store')  # macos dodaje ten plik do kazdego folderu
    images = sorted(images)
    fig, axes = plt.subplots(2, 4)
    axes = axes.flatten()
    for idx, img_file in enumerate(images):
        if idx >= len(axes):
            break
        img = mimg.imread(os.path.join(image_folder, img_file))
        axes[idx].imshow(img)
        axes[idx].set_title(img_file.split('.')[0])
        axes[idx].axis('off')
    plt.show()


def load_pokedex(description_file, image_folder):
    pokedex = pd.read_csv(description_file)
    # sortujemy pokedex alfabetycznie
    pokedex.sort_values(by=['Name'], ascending=True, inplace=True)
    pokedex.drop('Type2', axis=1, inplace=True)
    images = os.listdir(image_folder)
    images.remove('.DS_Store')  # macos dodaje ten plik do kazdego folderu
    images = sorted(images)
    images = list(map(lambda image_file: os.path.join(image_folder, image_file), images))

    pokedex['Image'] = images
    return pokedex


# ========================   SAVE    =========================
def save_model_to_json(model):
    json_model = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(json_model)


def save_frame_to_csv(frame):
    frame.to_csv('frame.csv', index=True)


# ========================   NETWORK    =========================
def prepare_data_for_network(pokedex):
    data_generator = ImageDataGenerator(validation_split=0.1,
                                        rescale=1.0 / 255,
                                        rotation_range=30,
                                        width_shift_range=0.5,
                                        height_shift_range=0.5,
                                        zoom_range=0.5, fill_mode='nearest')

    test_generator = ImageDataGenerator(validation_split=0.1,
                                        rescale=1.0 / 255)
    train_generator = data_generator.flow_from_dataframe(
        pokedex,
        x_col='Image',
        y_col='Type1',
        subset='training',
        color_mode='rgba',
        class_mode='categorical',
        target_size=(120, 120),
        shuffle=True,
        batch_size=32)

    test_generator = test_generator.flow_from_dataframe(
        pokedex,
        x_col='Image',
        y_col='Type1',
        subset='validation',
        color_mode='rgba',
        class_mode='categorical',
        target_size=(120, 120),
        shuffle=True)
    return train_generator, test_generator


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-d', '--description_file',
                        default='pokemon_dataset/pokemon.csv',
                        help='A CSV file with pokemon information')
    parser.add_argument('-i', '--image_folder',
                        default='pokemon_dataset/images',
                        help='Folder with pokemon images')
    return parser.parse_args()


def show_generator_results(generator):
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        for x, y in generator:
            img = x[0]
            plt.imshow(img)
            break
    plt.show()


def prepare_network():
    model = ks.models.Sequential()
    model.add(ks.layers.Conv2D(34, (3, 3), activation='relu', input_shape=(120, 120, 4)))
    model.add(ks.layers.MaxPooling2D(2, 2))
    model.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(ks.layers.MaxPooling2D(2, 2))
    model.add(ks.layers.Flatten())
    model.add(ks.layers.Dense(128, activation='relu'))
    model.add(ks.layers.Dense(18, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    # print(model.summary())
    return model


def main():
    args = parse_arguments()
    pokedex = load_pokedex(args.description_file, args.image_folder)

    # show_dataset_info(pokedex)
    # show_example_images(args.image_folder)

    train, test = prepare_data_for_network(pokedex)
    model = prepare_network()

    # show_generator_results(train)
    history = model.fit_generator(train, validation_data=test, epochs=20)  # zmienic se epoki
    # plt.plot(history.history['acc'])
    # plt.show()

    # dokładność
    hist_frame = pd.DataFrame(history.history)
    hist_frame.loc[:, ['acc', 'val_acc']].plot()  # nie dziala w riderze, test dla jupitera
    plt.show()


    # save:
    save_frame_to_csv(hist_frame)
    save_model_to_json(model)


if __name__ == '__main__':
    main()
