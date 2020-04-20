import scipy.sparse as sp
import numpy as np


def load_rating_file_as_list(path='ml-1m/'):
    ratingList = []
    with open(path+'ml-1m.test.rating', "r", encoding="utf-8") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item = int(arr[0]), int(arr[1])
            ratingList.append([user, item])
            line = f.readline()
    return ratingList


def load_negative_file(path='ml-1m/'):
    negativeList = []
    with open(path+'ml-1m.test.negative', "r", encoding="utf-8") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            negatives = []
            for x in arr[1:]:
                negatives.append(int(x))
            negativeList.append(negatives)
            line = f.readline()
    return negativeList


def load_rating_train_as_matrix(path='ml-1m/'):
        # Get number of users and items
    num_users, num_items = 0, 0
    with open(path+"u.info", "r", encoding="utf-8") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split(" ")
            if(arr[1].replace("\n", "") == 'users'):
                num_users = int(arr[0])
            if (arr[1].replace("\n", "") == 'items'):
                num_items = int(arr[0])
            line = f.readline()

    # Construct matrix
    mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
    with open(path+'ml-1m.train.rating', "r", encoding="utf-8") as f:
        line = f.readline()
        while line != None and line != "":
            arr = line.split("\t")
            user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
            if (rating > 0):
                mat[user, item] = 1.0
            line = f.readline()
    return mat


def load_itemGenres_as_matrix(path='ml-1m/'):
    num_items, num_type, dict = 0, 0, {}
    with open(path+"u.info", encoding="utf-8") as f:
        line = f.readline().strip('\n')
        while line != None and line != "":
            arr = line.split(" ")
            if (arr[1] == 'items'):
                num_items = int(arr[0])
            line = f.readline().strip('\n')

    with open(path+"u.genre", "r", encoding="utf-8") as f:
        line = f.readline().strip('\n')
        while line != None and line != "":
            arr = line.split("|")
            dict[arr[0]] = num_type
            num_type = num_type + 1
            line = f.readline().strip('\n')

    # Construct matrix
    mat = sp.dok_matrix((num_items + 1, num_type), dtype=np.float32)
    with open(path+"movies.dat", encoding="utf-8") as f:
        line = f.readline().strip('\r\n')
        while line != None and line != "":
            arr = line.split("::")
            types = arr[2].split("|")
            for ts in types:
                if(ts in dict.keys()):
                    mat[int(arr[0]), dict[ts]] = 1
            line = f.readline().strip('\r\n')
    itemGenres_mat = mat.toarray()

    return num_items, itemGenres_mat


def load_user_attributes(path='ml-1m/', split=False):
    usersAttributes = []
    num_users = 0
    dictGender = {}
    genderTypes = 0
    dictAge = {}
    ageTypes = 0
    dictOccupation = {}
    ocTypes = 0

    with open(path+'users.dat', "r", encoding="utf-8") as f:
        line = f.readline().strip('\n')
        while line != None and line != "":
            arr = line.split('::')
            l = []
            for x in arr[0:4]:
                l.append(x)
            usersAttributes.append(l)
            line = f.readline().strip('\n')
    usersAttrMat = np.array(usersAttributes)

    num_users = len(usersAttrMat)

    # one-hot encoder

    # age types

    genders = set(usersAttrMat[:, 1])
    for gender in genders:
        dictGender[gender] = genderTypes
        genderTypes += 1

    # age types
    ages = set(usersAttrMat[:, 2])
    for age in ages:
        dictAge[age] = ageTypes
        ageTypes += 1

    # occupation types
    ocs = set(usersAttrMat[:, 3])
    for oc in ocs:
        dictOccupation[oc] = ocTypes
        ocTypes += 1

    # Gender,Age,Occupation
    gendermat = sp.dok_matrix((num_users + 1, genderTypes), dtype=np.float32)
    agemat = sp.dok_matrix((num_users + 1, ageTypes), dtype=np.float32)
    occupationmat = sp.dok_matrix((num_users + 1, ocTypes), dtype=np.float32)

    with open(path+"users.dat", encoding="utf-8") as f:
        line = f.readline().strip('\n')
        while line != None and line != "":
            arr = line.split("::")
            userid = int(arr[0])
            usergender = arr[1]
            userage = arr[2]
            useroc = arr[3]
            # gender encoder
            if usergender in dictGender.keys():
                gendermat[userid, dictGender[usergender]] = 1.0
            # age encoder
            if userage in dictAge.keys():
                agemat[userid, dictAge[userage]] = 1.0
            # occupation encoder
            if useroc in dictOccupation.keys():
                occupationmat[userid, dictOccupation[useroc]] = 1.0

            line = f.readline().strip('\n')

    # add this line to original code to return the single matrixes (Simone Boglio)
    if split:
        return num_users, gendermat, agemat, occupationmat

    user_gender_mat = gendermat.toarray()
    user_age_mat = agemat.toarray()
    user_oc_mat = occupationmat.toarray()

    # concatenate Gender[0-1], Age[], Occupation
    onehotUsers = np.hstack((user_gender_mat, user_age_mat, user_oc_mat))
    return num_users, onehotUsers


def load_user_vectors():
    userNeighbors = open('neighbors/interNeighbors_20.txt', encoding="utf-8").readlines()
    #userNeighbors = open('neighbors/hammingNeighbors_20.txt').readlines()

    userVecmat = [[0] * 20]
    for u in userNeighbors:
        u = u.strip('\n')
        nbs = u.split('\t')
        #nbs = u.split(' ')
        userVecmat.append(nbs[0:20])
    mat = np.array(userVecmat)
    return mat
