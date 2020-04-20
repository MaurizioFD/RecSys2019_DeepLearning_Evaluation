#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/01/18

@author: Maurizio Ferrari Dacrema
"""


import ast, gzip, os
from Data_manager.Dataset import Dataset
from Data_manager.DataReader import DataReader
from Data_manager.DataReader_utils import download_from_URL, remove_features, load_CSV_into_SparseBuilder



def parse_json(file_path):
    g = open(file_path, 'r')

    for l in g:
        try:
            yield ast.literal_eval(l)
        except Exception as exception:
            print("Exception: {}. Skipping".format(str(exception)))



class _AmazonReviewDataReader(DataReader):

    DATASET_SUBFOLDER = "AmazonReviewData/"

    IS_IMPLICIT = False


    def _get_ICM_metadata_path(self, data_folder, compressed_file_name, decompressed_file_name, file_url):
        """
        Metadata files are .csv
        :param data_folder:
        :param file_name:
        :param file_url:
        :return:
        """


        try:

            open(data_folder + decompressed_file_name, "r")

        except FileNotFoundError:

            self._print("Decompressing metadata file...")

            try:

                decompressed_file = open(data_folder + decompressed_file_name, "wb")

                compressed_file = gzip.open(data_folder + compressed_file_name, "rb")
                decompressed_file.write(compressed_file.read())

                compressed_file.close()
                decompressed_file.close()

            except (FileNotFoundError, Exception):

                self._print("Unable to find or decompress compressed file. Downloading...")

                download_from_URL(file_url, data_folder, compressed_file_name)

                decompressed_file = open(data_folder + decompressed_file_name, "wb")

                compressed_file = gzip.open(data_folder + compressed_file_name, "rb")
                decompressed_file.write(compressed_file.read())

                compressed_file.close()
                decompressed_file.close()


        return data_folder + decompressed_file_name






    def _get_URM_review_path(self, data_folder, file_name, file_url):
        """
        Metadata files are .csv
        :param data_folder:
        :param file_name:
        :param file_url:
        :return:
        """


        try:

            open(data_folder + file_name, "r")

        except FileNotFoundError:

            self._print("Unable to find or open review file. Downloading...")
            download_from_URL(file_url, data_folder, file_name)


        return data_folder + file_name



    def _load_from_original_file_all_amazon_datasets(self, URM_path, metadata_path = None, reviews_path = None):
        # Load data from original


        self._print("loading URM")
        URM_all, URM_timestamp, self.item_original_ID_to_index, self.user_original_ID_to_index = load_CSV_into_SparseBuilder(URM_path, separator=",", header = False, timestamp = True)

        loaded_URM_dict = {"URM_all": URM_all,
                           "URM_timestamp": URM_timestamp}

        loaded_ICM_dict = {}
        loaded_ICM_mapper_dict = {}

        if metadata_path is not None:
            self._print("loading metadata")
            ICM_metadata, tokenToFeatureMapper_ICM_metadata, _ = self._loadMetadata(metadata_path, if_new_item ="ignore")

            ICM_metadata, _, tokenToFeatureMapper_ICM_metadata = remove_features(ICM_metadata, min_occurrence= 5, max_percentage_occurrence= 0.30,
                                                                                 reconcile_mapper=tokenToFeatureMapper_ICM_metadata)

            loaded_ICM_dict["ICM_metadata"] = ICM_metadata
            loaded_ICM_mapper_dict["ICM_metadata"] = tokenToFeatureMapper_ICM_metadata


        if reviews_path is not None:
            self._print("loading reviews")
            ICM_reviews, tokenToFeatureMapper_ICM_reviews, _ = self._loadReviews(reviews_path, if_new_item ="ignore")

            ICM_reviews, _, tokenToFeatureMapper_ICM_reviews = remove_features(ICM_reviews, min_occurrence= 5, max_percentage_occurrence= 0.30,
                                                                               reconcile_mapper=tokenToFeatureMapper_ICM_reviews)

            loaded_ICM_dict["ICM_reviews"] = ICM_reviews
            loaded_ICM_mapper_dict["ICM_reviews"] = tokenToFeatureMapper_ICM_reviews


        loaded_dataset = Dataset(dataset_name = self._get_dataset_name(),
                                 URM_dictionary = loaded_URM_dict,
                                 ICM_dictionary = loaded_ICM_dict,
                                 ICM_feature_mapper_dictionary = loaded_ICM_mapper_dict,
                                 UCM_dictionary = None,
                                 UCM_feature_mapper_dictionary = None,
                                 user_original_ID_to_index= self.user_original_ID_to_index,
                                 item_original_ID_to_index= self.item_original_ID_to_index,
                                 is_implicit = self.IS_IMPLICIT,
                                 )


        # Clean temp files
        self._print("cleaning temporary files")

        if metadata_path is not None:
            os.remove(metadata_path)

        if reviews_path is not None:
            os.remove(reviews_path)

        self._print("loading complete")

        return loaded_dataset







    def _loadMetadata(self, file_path, if_new_item = "ignore"):


        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = self.item_original_ID_to_index, on_new_row = if_new_item)


        from Data_manager.TagPreprocessing import tagFilterAndStemming, tagFilter
        import itertools


        parser_metadata = parse_json(file_path)

        numMetadataParsed = 0

        for newMetadata in parser_metadata:

            numMetadataParsed+=1
            if (numMetadataParsed % 20000 == 0):
                print("Processed {}".format(numMetadataParsed))

            item_ID = newMetadata["asin"]

            # The file might contain other elements, restrict to
            # Those in the URM

            tokenList = []

            #item_price = newMetadata["price"]

            if "title" in newMetadata:
                item_name = newMetadata["title"]
                tokenList.append(item_name)

            # Sometimes brand is not present
            if "brand" in newMetadata:
                item_brand = newMetadata["brand"]
                tokenList.append(item_brand)

            # Categories are a list of lists. Unclear whether only the first element contains data or not
            if "categories" in newMetadata:
                item_categories = newMetadata["categories"]
                item_categories = list(itertools.chain.from_iterable(item_categories))
                tokenList.extend(item_categories)


            if "description" in newMetadata:
                item_description = newMetadata["description"]
                tokenList.append(item_description)


            tokenList = ' '.join(tokenList)

            # Remove non alphabetical character and split on spaces
            tokenList = tagFilterAndStemming(tokenList)

            # Remove duplicates
            tokenList = list(set(tokenList))

            ICM_builder.add_single_row(item_ID, tokenList, data=1.0)


        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()







    def _loadReviews(self, file_path, if_new_item = "add"):


        from Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

        ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = None, on_new_col = "add",
                                                        preinitialized_row_mapper = self.item_original_ID_to_index, on_new_row = if_new_item)




        from Data_manager.TagPreprocessing import tagFilterAndStemming, tagFilter


        parser_reviews = parse_json(file_path)

        numReviewParsed = 0

        for newReview in parser_reviews:

            numReviewParsed+=1
            if (numReviewParsed % 20000 == 0):
                print("Processed {} reviews".format(numReviewParsed))

            user_ID = newReview["reviewerID"]
            item_ID = newReview["asin"]

            reviewText = newReview["reviewText"]
            reviewSummary = newReview["summary"]

            tagList = ' '.join([reviewText, reviewSummary])

            # Remove non alphabetical character and split on spaces
            tagList = tagFilterAndStemming(tagList)

            ICM_builder.add_single_row(item_ID, tagList, data=1.0)




        return ICM_builder.get_SparseMatrix(), ICM_builder.get_column_token_to_id_mapper(), ICM_builder.get_row_token_to_id_mapper()

