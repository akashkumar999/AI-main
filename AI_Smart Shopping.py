import os
import cv2
import pandas as pd
import numpy as np
from sklearn.cluster import MeanShift

import matplotlib.pyplot as plt
from collections import Counter

from tkinter import *
from tkinter import ttk

class ShoppingAssistant:
    def __init__(self, master):
        self.master = master
        master.title("Shopping Assistant")

        self.label = Label(master, text="Enter a keyword to search:")
        self.label.pack()

        self.keyword_entry = Entry(master, width=50)
        self.keyword_entry.pack()

        self.search_button = Button(master, text="Search", command=self.search)
        self.search_button.pack()

        self.result_label = Label(master, text="")
        self.result_label.pack()

    def search(self):
        keyword = self.keyword_entry.get()
        PATH = "data/"

        for category in os.listdir(PATH):
            if category == keyword:
                path = os.path.join(PATH, category)
                for image in os.listdir(path):
                    img = cv2.imread(os.path.join(path, image))
                    img = cv2.resize(img, (200, 200))
                    cv2.imshow(category, img)
                    cv2.waitKey(1000)

        centers = [[1, 1], [5, 5], [8, 4]]

        dataset = pd.read_csv('person.csv')

        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, [0]].values
        name = dataset['Item_names'].tolist()

        ms = MeanShift()
        ms.fit(X)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_

        gg = Counter(labels)

        def find_max():
            max = gg[0]
            v = 0
            for i in range(len(gg)):
                if gg[i] > max:
                    max = gg[i]
                    v = i
            return v

        Y = y.tolist()
        L = labels.tolist()

        max_label = find_max()

        suggest = []
        for i in range(len(labels)):
            if max_label == L[i]:
                suggest.append(Y[i])

        new = []

        def stripp(rr):
            for i in range(len(suggest)):
                p = str(rr[i]).replace('[', '').replace(']', '')
                new.append(int(p))
            return new

        new_Y = stripp(Y)
        new_name = []
        for i in range(len(suggest)):
            p = str(name[i]).replace('[', '').replace(']', '')
            new_name.append(p)
 
        n_clusters_ = len(np.unique(labels))

        suggest = 10
        colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']

        for i in range(len(X)):
            plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10)

        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker="x", s=150, linewidths=5, zorder=10)

        item_name = dict(zip(new_Y, new_name))

        result = "Recommendations:\n"
        for i in range(suggest):
            result += "Item ID- {}   Item name- {}\n".format(new_Y[i], new_name[i])

        self.result_label.config(text=result)
        
        plt.show()


root = Tk()
my_gui = ShoppingAssistant(root)
root.mainloop()
