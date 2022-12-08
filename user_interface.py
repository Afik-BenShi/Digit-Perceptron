from net import Network
import mnist_loader
import image_processing
import os
import warnings


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
            1. Upload an image to be detected by a trained network
            2. Create your own new network'''
    
    print(welcome)
    while True:
        print('\n\nPlease choose one option from the menu below:')
        print(main_menu)
        choice = input('\n\ninput the number of your choice ')
        if choice == '1':
            clearConsole()
            print('Image detection!\n')
            N = Network.load_from_json()
            input("Please select your image \nPressENTER to continue")
            img = image_processing.load_image()
            N.detect(img,detailed=True)
            input('Press ENTER to return to main menu')
        elif choice == '2':
            clearConsole()
            print('sorry, not ready yet')
        else:
            clearConsole()
            print(f'choice is not in the menu.')

if __name__ == '__main__':
    main()