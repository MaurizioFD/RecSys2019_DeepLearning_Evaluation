
Descriptions
DATASET DESCRIPTION

Ta-Feng is a grocery shopping dataset released by ACM RecSys, it covers products from food, office supplies to furniture.
The dataset collected users` transaction data of 4 months, from November 2000 to February 2001. The total count of transactions in this dataset is 817741, which belong to 32266 users and 23812 products. See http://recsyswiki.com/wiki/Grocery_shopping_datasets for more details.
FILE DESCRIPTION
D.txt

The file D.txt records users` transaction history. Each line in the file correspond to a transaction in the following format
Transaction date;customerID;Age group; Residence Area; Product subclass;Amount;Asset;Sales Price;

1: Transaction date and time (time invalid and useless)
2: Customer ID
3: Age: 10 possible values,
A <25,B 25-29,C 30-34,D 35-39,E 40-44,F 45-49,G 50-54,H 55-59,I 60-64,J >65
4: Residence Area: 8 possible values,
A-F: zipcode area: 105,106,110,114,115,221,G: others, H: Unknown
Distance to store, from the closest: 115,221,114,105,106,110
5: Product subclass
6: Product ID
7: Amount
8: Asset
9: Sales price
train.txt and user_tran

File train.txt is the training set, it contains all userstransaction data in D except the last transaction of each user. user_tran is a transformation of train.txt, each line of it records additional information of users next transaction. The line is formatted as follows:
userid, tranid, product id list in the tran , one product id in the next transaction of user
test.txt

File test.txt is the testing set, which contains the last transaction of each user. Detail of each line is:
indexID of transaction, product id list in this transaction, userID, transaction date.
