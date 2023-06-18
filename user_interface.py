from net import Network
import mnist_loader
import image_processing
import os
import warnings
import random
from glob import glob

YES = {'Y','y','YES','yes','Yes'}
NO = {'N','n','NO','no','No'}
N = None

def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)


def main():
    warnings.filterwarnings("ignore")
    welcome = '''       
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            |Welcome to My Mediocre Perceptron|
            ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    This program will allow you to use a neural network to read actual digits!
    You can use a pretrained network or train one for yourself.'''
    main_menu = '''
                        MENU
                        ~~~~
            1. Upload an image to be detected
            2. Detect a random image from the testing database
            3. Create your own new network
            4. Load a specific network'''
    print('Initializing...')
    N = Network.load_from_json()
    test_d = mnist_loader.load_test_data()
    clearConsole()
    print(welcome)
    while True:
        print('\n\n')
        print(main_menu)
        choice = input('\n\ninput the number of your choice or exit to exit ')
        if choice == '1':
            clearConsole()
            print('Image detection!\n')
            N = Network.load_from_json()
            input("A window will open for you to select your image \nPress ENTER to select")
            img = image_processing.load_image()
            N.detect(img,detailed=True)
            input('Press ENTER to return to main menu')
        elif choice == '2':
            clearConsole()
            while True:
                clearConsole()
                print("A random image will appear and then the network's guess")
                image = random.choice(test_d)
                N.detect(image, detailed=True)
                again = input("again? (y/n) ")
                if again in YES:
                    continue
                break
        elif choice == '3':
            clearConsole()
            print('Choose number of neurons in each layer, seperated by commas')
            input_sizes = input('[').split(',')
            print(']')
            sizes = [None for _ in range(len(input_sizes))]
            for i, size in enumerate(input_sizes):
                if size.isnumeric():
                    num = int(size)
                else:
                    break
                sizes[i] = num
            # input assertion
            if len(input_sizes) <= 1:
                print('incorrect input')
                continue
            print("Loading training datasets...", end="")
            N = Network(sizes)
            train_d = mnist_loader.load_train_data()
            test_d = mnist_loader.load_test_data()
            print(" done")
            print('''The training is done using a dataset of 20,000 labels of handwritten numbers.
            When training, we split the dataset to batches, and adjust our parameters after each batch.
            After we go over the whole dataset, we may want to iterate over it again to improve our performance.
            Therefore we need to choose two parameters: the batch size and the number of repetitions (usually called epochs).
            ''')
            batches_input = input("Batch size (does not have to devide 20,000): ")
            while not batches_input.isnumeric():
                print("Batch size must be a whole number")
                batches_input = input("Batch size: ")
            epoch_input = input("Epochs: ")
            while not epoch_input.isnumeric():
                print("Number of repetitions must be a whole number")
                epoch_input = input("Epochs: ")
            clearConsole()
            print("######## Training ########")
            retrain = 'y'
            while retrain in YES:
                N.SGD(train_d, int(epoch_input), int(batches_input), 3.0, test_d, 'epochs')
                retrain = input('Want to train once more? (y/n) ')
            clearConsole()
            print("######## Training  done! ########")
            print("any operation you'll do next will be using your trained network.")
            save = input("Do you want to save your network to file? (y/n) ")
            if save in YES:
                savefile = N.save_to_json()
                print(f'Your network is saved as: "{savefile}"')
                
        elif choice == '4':
            filenames = glob('saves/*.json')
            for i, file in enumerate(filenames):
                print(f'{i+1}. {file.split("/")[-1][:-5]}')
            filechoice = input("insert the number of the network you want to choose ")
            while not filechoice.isnumeric() or int(filechoice) > len(filenames) or int(filechoice) <= 0:
                print("Choise is not in the options")
                filechoice = input("insert the number of the network you want to choose ")
            loaded = Network.load_from_json(filenames[int(filechoice)-1])
            clearConsole()
            if loaded:
                print('Network selected.')
                N = loaded
            else:
                print('There was a problem, please try again.')

        elif choice == 'exit':
            print("goodbye")
            break
        else:
            clearConsole()
            print(f'choice is not in the menu.')

if __name__ == '__main__':
    main()