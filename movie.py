# run command as example
# python movie.py --n_users 100 --n_movies 1000 --n_ratings 200 --min_rating 1 --max_rating 5

import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Creating Movie Dataset')
parser.add_argument('--n_users', type=int, help='Number of Users')
parser.add_argument('--n_movies', type=int, help='Number of movies')
parser.add_argument('--n_ratings', type=int, help='Number of watched movies by each user')
parser.add_argument('--min_rating', type=int, help='Minimum rating given by user')
parser.add_argument('--max_rating', type=int, help='Maximum rating given by user')

args = parser.parse_args()

class movie_dataset:
    def __init__(self, n_users, n_movies, n_ratings, min_rating, max_rating):
        
        self.n_users = n_users
        self.n_movies = n_movies
        self.n_ratings = n_ratings
        self.min_rating = min_rating
        self.max_rating = max_rating
        
    def make_ground_truth(self): 
        return np.array(list(map(lambda x: np.random.choice(np.arange(self.n_movies), 
                                                            self.n_ratings, replace=False), np.arange(self.n_users))))
    
    def make_ratings(self):
        
        return np.array(list(map(lambda x: np.random.choice(np.arange(self.min_rating, self.max_rating+1), 
                                                            self.n_ratings), np.arange(self.n_users))))
    
    def make_predictions(self):
        gt = self.make_ground_truth()
        
        movie_ids = list(range(self.n_movies))
        movie_ids = np.array([movie_ids]*self.n_users)

        return np.array(list(map(lambda x,y: self.find_unrated_movies(x,y), movie_ids, gt)))        
    
    
    def find_unrated_movies(self, movies, watched):
        np.random.shuffle(movies)

        return list(set(movies)-set(watched))


if __name__=='__main__':
    
    dataset = movie_dataset(args.n_users, args.n_movies, args.n_ratings, args.min_rating, args.max_rating)

    #movies watched by users
    gt = dataset.make_ground_truth()

    #ratings given by users to watched movies
    gt_ratings = dataset.make_ratings()

    #predictions made for users
    predictions = dataset.make_predictions()
    
    print(gt.shape), print(gt_ratings.shape), print(predictions.shape)
    