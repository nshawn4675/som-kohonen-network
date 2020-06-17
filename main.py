import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler

class SOM():
    def __init__(self, x_len, y_len, in_dim):
        self.x_len = x_len
        self.y_len = y_len
        self.in_dim = in_dim
        self.w = self.init_weights()
        
    def init_weights(self):
        w = np.random.random((self.x_len, self.y_len, self.in_dim))
        for x in range(self.x_len):
            for y in range(self.y_len):
                w[x][y] /= np.linalg.norm(w[x][y], axis=0, keepdims=True)
        return w
        
    def train(self, features, iteration, lr=0.7):
        self.iter = iteration
        for cur_iter in range(self.iter):
            for feature in features:
                winner = self.find_winner(feature)
                neighbor_r = self.cal_neighbor_r(cur_iter)
                neighbors = self.get_neighbors_list(winner, neighbor_r)
                L = self.cal_lr(lr, cur_iter)
                for neighbor in neighbors:
                    theta = self.cal_theta(winner, neighbor, neighbor_r)
                    self.update_weight(L, theta, feature, neighbor)
            print(f'training process : {cur_iter+1} / {self.iter}')
                
    def find_winner(self, feature):
        winner_x = -1
        winner_y = -1
        min_vec_dist = math.inf
        for x in range(self.x_len):
            for y in range(self.y_len):
                vec_dist = self.cal_vec_dist(feature, self.w[x][y])
                if vec_dist < min_vec_dist:
                    min_vec_dist = vec_dist
                    winner_x = x
                    winner_y = y
        #print(f'winner = {winner_x}, {winner_y}')
        return tuple([winner_x, winner_y])
    
    def cal_vec_dist(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))
    
    def cal_neighbor_r(self, cur_iter):
        lbd = self.iter / (math.log(self.x_len))
        sigma = self.x_len * (math.exp(-(cur_iter / lbd)))
        sigma = round(sigma)
        #print(f'neighbor_r = {sigma}')
        return sigma
    
    def get_neighbors_list(self, winner, r):
        neighbors = []
        for x in range(self.x_len):
            for y in range(self.y_len):
                neighbor_dist = (winner[0]-x)**2 + (winner[1]-y)**2
                neighbor_dist=math.sqrt(neighbor_dist)
                if neighbor_dist < r:
                    neighbors.append(tuple([x, y]))
        return neighbors
                    
    def cal_lr(self, lr, cur_iter):
        lbd = self.iter / math.log(lr)
        L = lr * (math.exp(-(cur_iter / lbd)))
        #print(f'L = {L}')
        return L
    
    def cal_theta(self, winner, neighbor, neighbor_r):
        dist = np.sqrt(np.sum(np.square(self.w[winner[0]][winner[1]]
                                        -self.w[neighbor[0]][neighbor[1]])))
        dist = dist**2
        neighbor_r = neighbor_r**2
        #print(f'dist={dist}, neighbor_r={neighbor_r}')
        theta = math.exp(-(dist/2*neighbor_r))
        #print(f'theta = {theta}')
        return theta
                    
    def update_weight(self, L, theta, feature, neighbor):
        self.w[neighbor[0]][neighbor[1]] += \
            L*theta*(feature - self.w[neighbor[0]][neighbor[1]])
    
    def plot_features(self, labels, features):
        occupy_matrix = [[-1 for y in range(self.y_len)] 
                         for x in range(self.x_len)]
        data_len = len(labels)
        colors = {1: "red", 2: "green", 3: "blue"}
        res = {1: [], 2: [], 3: []}
        plt.xlim(0,15)
        plt.ylim(0,15)
        plt.title("result")
        for i in range(data_len):
            min_dist = math.inf
            winner = [-1, -1]
            for x in range(self.x_len):
                for y in range(self.y_len):
                    if occupy_matrix[x][y] == -1:
                        vec_dist = self.cal_vec_dist(features[i], self.w[x][y])
                        if vec_dist < min_dist:
                            winner = [x, y]
                            min_dist = vec_dist
            occupy_matrix[winner[0]][winner[1]] = 1
            res[labels[i]].append(winner)
            print(f'plot result : {i+1} / {data_len}')
        for i in range(1, 3+1):
            plt.scatter([xy[0] for xy in res[i]], [xy[1] for xy in res[i]],
                        color=colors[i], label=str(i))
        plt.legend(loc="upper right")
        plt.show()
                    

def get_data_label_from(file_name):
    df = pd.read_csv('wine.data')
    features = df.iloc[:, 1:].values
    features = MinMaxScaler(feature_range = (0, 1)).fit_transform(features)
    # print(f'features={features}')
    features_list = features.tolist()
    label_list = df.iloc[:, 0].values.tolist()
    features, labels = features_list, label_list
    return features, labels

if __name__ == '__main__':
    features, labels = get_data_label_from('wine.data')
    som = SOM(x_len=15, y_len=15, in_dim=len(features[0]))
    som.train(features, iteration=100, lr=0.7)
    som.plot_features(labels, features)
