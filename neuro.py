import scipy.special
import matplotlib.pyplot
import imageio
import glob


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать количество узлов во входном, скрытом и выходном слое
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # коэффициент обучения
        self.lr = learningrate
        '''self.wih = (numpy.random.rand(self.hnodes, self.inodes) - 0.5)
        self.who = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        '''
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs *(1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # обновить весовые коэффициенты связей между скрытым и входным слоями
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs *(1.0 - hidden_outputs)), numpy.transpose(inputs))

    # опрос нейронной сети
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inp = numpy.dot(self.wih, inputs)
        hidden_outp = self.activation_function(hidden_inp)

        final_inp = numpy.dot(self.who, hidden_outp)

        final_outp = self.activation_function(final_inp)

        return final_outp


def training():
    print('Введите коэффициент обучения')
    learning_rate = float(input('Коэффициент = '))

    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    print('Введите количество эпох обучения')
    epochs = input('Количество эпох = ')
    ans = True
    while ans:
        if epochs.isnumeric():
            epochs = int(epochs)
            ans = False
        else:
            print('Ошибка! Введите число')
            epochs = input()
    train_dataset = []
    train_labels = []

    for image_file_name in glob.glob('train_?.png'):
        label = image_file_name[-5:-4]
        img_data = 0

        print("Загрузка изображения ... ", image_file_name)
        img_array = imageio.imread(image_file_name, as_gray=True)

        img_data = 255.0 - img_array.reshape(784)

        img_data = (img_data / 255.0 * 0.99) + 0.01

        train_dataset.append(img_data)
        train_labels.append(label)
        pass

    for e in range(0, epochs):
        for item in range(0, len(train_dataset)):
            # plot image
            # matplotlib.pyplot.imshow(train_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

            # correct answer is first value
            # correct_label = train_dataset[item][0]
            correct_label = train_labels[item]
            # data is remaining values
            # inputs = train_dataset[item][1:]
            inputs = train_dataset[item]
            index = 0
            for oper in range(0, len(corresponding_image)):
                if correct_label == corresponding_image[oper]:
                    index = oper

            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record
            targets[index] = 0.99
            # query the network
            n.train(inputs, targets)
    return n


def test(n):
    print('Введите название изображения для тестирования (пример my_own_images/test_&.png)')
    image_file_name = input('Имя - ')
    # use the filename to set the correct label
    label = image_file_name[-5:-4]
    # load image data from png files into an array
    print("Загрузка изображения ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)

    # reshape from 28x28 to list of 784 values, invert values
    img_data = 255.0 - img_array.reshape(784)

    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    # print(numpy.min(img_data))
    # print(numpy.max(img_data))

    # append label and image data  to test data set
    record = numpy.append(label, img_data)
    test_dataset = record
    # test the neural network with our own images

    # plot image
    # matplotlib.pyplot.imshow(test_dataset[0][1:].reshape(28,28), cmap='Greys', interpolation='None')

    # correct answer is first value
    correct_label = test_dataset[0]
    # data is remaining values
    inputs = img_data

    # query the network
    outputs = n.query(inputs)
    # print (outputs)

    # the index of the highest value corresponds to the label
    lab = numpy.argmax(outputs)
    sum = 0
    for i in outputs:
        sum += i
    print('Вероятность',(outputs[lab]/sum)*100,'%')
    label = corresponding_image[lab]
    print("Отклик системы ", label)
    # append correct or incorrect to list
    if (label == correct_label):
        print("Совпадение!")
    else:
        print("Нет совпадений!")
        pass

corresponding_image = ['&','↑','↓','↔','∨','∧', '→', '￢', '⊕','~']
print('Распознаваемые образы: ', ', '.join(corresponding_image))
# количество входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
# коэффициент обучения


ans1 = 'Да'

n = []
while ans1=='Да':
    print('Выберете режим. Т - тестирование, О - обучение')
    mode = input()
    mode = mode.lower().capitalize()
    if mode =='О':
        n = training()
    if mode == 'Т':
        if not n:
            print('Для начала необходимо обучить систему')
        else:
            test(n)
    print('Хотите ли выбрать другой режим?(Да, Нет)')
    ans1 = input()
    ans1 = ans1.lower().capitalize()
