class DataContainer:

    def __init__(self):
        self.train = []  # Numpy array containing all the images [images, w, h, d]
        self.train_labels = []  # Numpy array containing all the classes [images, number of classes]
        self.classes = []
        self.totClasses = 0  # Number of classes in our data
        self.size = []  # Width, Height, Depth
        self.test= []
        self.test_labels = []
