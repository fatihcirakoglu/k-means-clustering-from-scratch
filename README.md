# SWE-584-Exploratory Data Analysis and Visualization
# Design of Standard K-Means Clustering Algorithm From Scratch

## Project Scope:
Functional Standard K-Means algorithm needs an input where you need to provide number of
clusters like k: 2,3,7. Purpose of standard K-Means algorithm does not aim to find how many
clusters there are in a data set, it is a very simple algorithm that you give number of clusters as input and it finds the best division between them.
## Observations:
Below pictures which are taken during observation shows the main differences on the shape of clusters and their point of centers while K-Means algorithm runs and improves the cluster divisioons.
### A-) Our own K-Means Algorithm Observations:
This is a 2D data set that includes 3 clusters and 3000 data samples. The delta value that will determine when algorithm will stop iterating is like:
epsilon = 0.0000001

![1](https://user-images.githubusercontent.com/33651899/176983655-0ae802fd-7f58-48ef-b8d1-30c2aec2f50d.jpg)


Figure-1 Visualization of Randomly Generated 2D Raw Data Set



![2](https://user-images.githubusercontent.com/33651899/176983656-f398cc1e-dee4-4138-97d5-f0464ef841ef.jpg)
Figure-2 Visualization of Cluster Centers in 1st iteration during our KMeans
Algorithm Works


As we see from Figure-2, initial centers of clusters are assigned randomly since they are very strange.

![3](https://user-images.githubusercontent.com/33651899/176983657-126ec406-fe70-49d2-83b4-f94a67f7747d.jpg)
Figure-3 Visualization of Cluster Centers in 2nd iteration during our KMeans
Algorithm works

As we see from Figure-3, after even after 2 iteration, centers of clusters are getting improved by the calculation of within cluster deviation for each group or cluster.


![4](https://user-images.githubusercontent.com/33651899/176983658-0eb3a102-52b7-408c-98de-5c1d323027aa.jpg)
Figure-4 Visualization of Cluster Centers in 3rd iteration during our KMeans
Algorithm works


As we see from Figure-4, after 3 iteration, centers of clusters are getting improved by the
calculation of within cluster deviation for each group or cluster. They are located better than 2nd iteration.


![5](https://user-images.githubusercontent.com/33651899/176983659-50a1e47f-105f-4778-b9fb-8256fdde3ce8.jpg)
Figure-5 Visualization of Cluster Centers in last iteration during our KMeans
Algorithm works

As we see from Figure-5, after when the clusters converge each other and enver changes at all after some iterations, centers of clusters are located perfectly by the calculation of within cluster deviation for each group or cluster in every iteration. So, this picture shows the best location of these clusters in this Data set.

### B-) Scikit K-Means Algorithm Observations:

![6](https://user-images.githubusercontent.com/33651899/176983660-6bab0d3f-bba5-4074-893d-a11bc0b57f8d.jpg)



Figure-6 Visualization of Clusters after Scikit K-Means Algorithm run

Figure-6 and Figure-7 shows the results while running original builtin Scikit K-Means algorithm on the same data set, since Scikit K-Means runs the algorithm at least 10 times and selects the best result from these set of results, its clustering is better than our own algorithm.



![7](https://user-images.githubusercontent.com/33651899/176983661-dc2a68d7-a165-4a21-9ab0-7702a932f620.jpg)


Figure-7 Visualization of Clusters Centers after Scikit K-Means Algorithm
run

### C-) Our own K-Means Algoritm Results when k=7: 

Data Set
![8](https://user-images.githubusercontent.com/33651899/176983662-1b9acf0f-3ac9-4a5a-a27e-d8eca3c7a66d.jpg)


First Iteration
![9](https://user-images.githubusercontent.com/33651899/176983663-6a0f9580-3a26-4dbe-ad45-0c2514e4972a.jpg)


Second Iteration
![10](https://user-images.githubusercontent.com/33651899/176983664-36f6328c-42b0-4d33-bd16-3876454fcca4.jpg)


Third Iteration
![11](https://user-images.githubusercontent.com/33651899/176983665-9a088542-3168-4c4f-a49f-82c5431fae89.jpg)


Final Form of Clustering when k=7
![12](https://user-images.githubusercontent.com/33651899/176983667-fbfec08f-c77a-4999-8a70-49745a449515.jpg)



### D-) Scikit K-Means Algoritm Results when k=7:

![13](https://user-images.githubusercontent.com/33651899/176983668-c1d3c70f-26d6-40d5-9e2b-a8a2dcfa2ccb.jpg)


![14](https://user-images.githubusercontent.com/33651899/176983669-86e526fc-b1e1-4eb5-8902-f4265b313f9f.jpg)

## Conclusion:
When we change the cluster number k upto 7, our algorithm has still worked but in both cases, scikit K-Means has much better results since it runs its own clustering algorithm at least 10 times and selects the best one and presents it. I couldn’t find enough time to plot objective function versus iterations, I will provide it separately.