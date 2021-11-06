import random
from bs4 import BeautifulSoup
from os.path import isfile
import json
import pandas as pd
import requests
from csv import writer

audience_review_pattern = "https://www.rottentomatoes.com/m/{}/reviews?type=user"

rt_movie_list_file = 'data/rt/rotten_tomatoes_movies.csv.zip'
movie_reviews_file = 'data/rt/movie_reviews.csv'

review_elem_class = 'audience-reviews__review-wrap'
review_text_class = 'audience-reviews__review'
review_score_class = 'star-display'
full_star_class = 'star-display__filled'
half_star_class = 'star-display__half'


def get_movie_reviews(movie_name: str) -> list:
    """ Returns a list of reviews for a given movie """
    url = audience_review_pattern.format(movie_name)
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    reviews = soup.find_all('div', class_=review_elem_class)
    if len(reviews) == 0:
        raise Exception('No reviews found for {}'.format(movie_name))
    for review in reviews:
        review_text = review.find('p', class_=review_text_class).text
        review_score_wrapper = review.find('span', class_=review_score_class)
        full_stars = review_score_wrapper.find_all('span',
                                                   class_=full_star_class)
        half_stars = review_score_wrapper.find_all('span',
                                                   class_=half_star_class)
        review_score = len(full_stars) + len(half_stars) / 2
        yield review_text, review_score


def get_movie_id_list() -> list:
    """ Returns a list of movies to process """
    df = pd.read_csv(rt_movie_list_file)

    movie_id_list = df['rotten_tomatoes_link'].squeeze().str.slice(
        start=2).tolist()
    return movie_id_list


def get_unique_movies_from_review_csv() -> list:
    """ Returns a list of movies that have been processed """
    if isfile(movie_reviews_file):
        df = pd.read_csv(movie_reviews_file, header=None)
        return df[0].unique().tolist()
    return []


if __name__ == '__main__':
    error_count = 0
    movie_id_list = get_movie_id_list()
    processed_movie_list = get_unique_movies_from_review_csv()
    random.shuffle(movie_id_list)
    total_num_movies = len(movie_id_list)
    for movie_id in movie_id_list:
        if movie_id in processed_movie_list:
            print('Movie {} already processed, skipping.'.format(movie_id))
            continue
        print('Scraping {}'.format(movie_id))
        reviews = get_movie_reviews(movie_id)
        processed_movie_list.append(movie_id)
        with open(movie_reviews_file, 'a+', newline='') as f:
            csv_writer = writer(f)
            n = 0
            try:
                for review in reviews:
                    n += 1
                    csv_writer.writerow([movie_id, review[0], review[1]])
            except Exception as e:
                error_count += 1
                print(e)
                if error_count > 10:
                    raise Exception(
                        'Too many sequential errors, aborting, most likely banned by RT.'
                    )
                continue
            # reset error count
            error_count = 0

        print('Saved {} reviews for {}. Total movies scraped: {}/{}'.format(
            n, movie_id, len(processed_movie_list), total_num_movies))
