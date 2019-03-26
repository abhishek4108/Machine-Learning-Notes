# Recommendation Engine

Provide the most relevant and accurate items to the user by filtering useful stuff from of a huge pool of information base. Recommendation engines discovers data patterns in the data set by learning consumers choices and produces the outcomes that co-relates to their needs and interests.

>[Analytics Vidhya - Recommendation Engines](https://www.analyticsvidhya.com/blog/2015/10/recommendation-engines/) <br />
[Comprehensive Guide - Python](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/)


# How a recommendation engine works?
<!-- toc -->

- [Recommendation Engine](#recommendation-engine)
- [How a recommendation engine works?](#how-a-recommendation-engine-works)
  - [1. Data Collection](#1-data-collection)
  - [2. Data Storage](#2-data-storage)
  - [3. Filtering](#3-filtering)
    - [3.1 Content Based Recommendations](#31-content-based-recommendations)
    - [3.2 Collaborative Filtering](#32-collaborative-filtering)
      - [3.2.1 User-User collaborative filtering](#321-user-user-collaborative-filtering)
      - [3.2.2 Item-Item collaborative filtering](#322-item-item-collaborative-filtering)

<!-- tocstop -->
## 1. Data Collection

- **Explicitly** - Explicit data is information that is provided intentionally, i.e. input from the users such as movie ratings. e.g. Netflix
- **Implicitly** - Implicit data is information that is not provided intentionally but gathered from available data streams like search history, clicks, order history, etc. e.g. Amazon

## 2. Data Storage

## 3. Filtering

After collecting and storing the data, we have to filter it so as to extract the relevant information required to make the final recommendations.

### 3.1 Content Based Recommendations

Content based systems, recommends item based on a similarity comparison between the content of the items and a user’s profile. The feature of items are mapped with feature of users in order to obtain user – item similarity.

Content-based filtering algorithm finds the cosine of the angle between the profile vector and item vector, i.e. cosine similarity. Suppose A is the profile vector and B is the item vector, then the similarity between them can be calculated as: <br />
$$
sim(A,B) = \cos(\theta) = \frac{A.B}{||A||||B||}
$$

Other methods that can be used to calculate the similarity are:

- **Euclidean Distance** -  Similar items will lie in close proximity to each other if plotted in n-dimensional space <br />
$$
ED = \sqrt{(x_1 - y_1)^2 + \cdots + (x_N - y_N)^2}
$$
- **Pearson’s Correlation** - It tells us how much two items are correlated. Higher the correlation, more will be the similarity. <br />
$$
sim(u,v) = \frac{\sum(r_{ui}- \overline r_u)(r_{vi}- \overline r_v)}{\sqrt{\sum(r_{ui}- \overline r_u)^2}\sqrt{\sum(r_{vi}- \overline r_v)^2}}
$$

A major drawback of this algorithm is that it is limited to recommending items that are of the same type. It will never recommend products which the user has not bought or liked in the past. It lacks in detecting inter dependencies or complex behaviors.

### 3.2 Collaborative Filtering

Collaborative Filtering algorithm considers “User Behaviour” for recommending items. They exploit behaviour of other users and items in terms of transaction history, ratings, selection and purchase information. Other users behaviour and preferences over the items are used to recommend items to the new users. In this case, features of the items are not known.

#### 3.2.1 User-User collaborative filtering
This algorithm first finds the similarity score between users. Based on this similarity score, it then picks out the most similar users and recommends products which these similar users have liked or bought previously.

The prediction of an item for a user u is calculated by computing the weighted sum of the user ratings given by other users to an item i.
The prediction Pu,i is given by: <br />
$$
P_{u,i} = \frac{\sum_v(r_{v,i}*s_{u,v})}{\sum_vs_{u,v}}
$$

Here,
- Pu,i is the prediction of an item
- Rv,i is the rating given by a user v to a movie i
- Su,v is the similarity between users

1. For predictions we need the similarity between the user u and v. We can make use of Pearson correlation.
2. First we find the items rated by both the users and based on the ratings, correlation between the users is calculated.
3. The predictions can be calculated using the similarity values. This algorithm, first of all calculates the similarity between each user and then based on each similarity calculates the predictions. Users having higher correlation will tend to be similar.
4. Based on these prediction values, recommendations are made. Let us understand it with an example:

This algorithm is quite time consuming as it involves calculating the similarity for each user and then calculating prediction for each similarity score. One way of handling this problem is to select only a few users (neighbors) instead of all to make predictions, i.e. instead of making predictions for all similarity values, we choose only few similarity values. There are various ways to select the neighbors:

- Select a threshold similarity and choose all the users above that value
- Randomly select the users
- Arrange the neighbors in descending order of their similarity value and choose top-N users
- Use clustering for choosing neighbors

This algorithm is useful when the number of users is less. Its not effective when there are a large number of users as it will take a lot of time to compute the similarity between all user pairs.


#### 3.2.2 Item-Item collaborative filtering

In this algorithm, we compute the similarity between each pair of items.
In movie case, we will find the similarity between each movie pair and based on that, we will recommend similar movies which are liked by the users in the past.
Instead of taking the weighted sum of ratings of “user-neighbors”, we take the weighted sum of ratings of “item-neighbors”.

The prediction Pu,i is given by: <br />
$$
P_{u,i} = \frac{\sum_v(s_{i,N}*R_{u,N})}{\sum_v(s_{i,N})}
$$

Now we will find the similarity between items. <br />
$$
sim(i,j) = \cos(\vec{i},\vec{j}) = \frac{\vec{i},\vec{j}}{||\vec{i}||_2*||\vec{j}||_2}
$$

What will happen if a new user or a new item is added in the dataset?
It is called a **Cold Start.** There can be two types of cold start:
1. Visitor Cold Start
2. Product Cold Start

In Visitor Cold Start, one basic approach could be to apply a popularity based strategy, i.e. recommend the most popular products. These can be determined by what has been popular recently overall or regionally.

On the other hand, in Product Cold Start we can make use of Content based filtering to solve this problem. The system first uses the content of the new product for recommendations and then eventually the user actions on that product.
