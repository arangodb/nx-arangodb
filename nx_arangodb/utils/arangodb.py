from arango import ArangoError, DocumentInsertError
from arango.database import StandardDatabase
from typing import Any


class ArangoDBBatchError(ArangoError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(self._format_errors())

    def _format_errors(self):
        return "\n".join(str(error) for error in self.errors)


def check_list_for_errors(lst):
    for element in lst:
        if element is type(bool):
            if element is False:
                return False

        elif isinstance(element, list):
            for sub_element in element:
                if isinstance(sub_element, DocumentInsertError):
                    return False

    return True


def extract_arangodb_key(arangodb_id):
    assert "/" in arangodb_id
    return arangodb_id.split("/")[1]


def extract_arangodb_collection_name(arangodb_id):
    assert "/" in arangodb_id
    return arangodb_id.split("/")[0]


def is_arangodb_id(key):
    return "/" in key


def get_arangodb_collection_key_tuple(key):
    assert is_arangodb_id(key)
    if is_arangodb_id(key):
        return key.split("/", 1)


def separate_nodes_by_collections(nodes: Any, default_collection: str) -> Any:
    """
    Separate the dictionary into collections based on whether keys contain '/'.

    :param nodes:
        The input dictionary with keys that may or may not contain '/'.
    :param default_collection:
        The name of the default collection for keys without '/'.
    :return: A dictionary where the keys are collection names and the
        values are dictionaries of key-value pairs belonging to those
        collections.
    """
    separated: Any = {}

    for key, value in nodes.items():
        if is_arangodb_id(key):
            collection, doc_key = get_arangodb_collection_key_tuple(key)
            if collection not in separated:
                separated[collection] = {}
            separated[collection][doc_key] = value
        else:
            if default_collection not in separated:
                separated[default_collection] = {}
            separated[default_collection][key] = value

    return separated


def transform_local_documents_for_adb(original_documents):
    """
    Transform original documents into a format suitable for UPSERT
    operations in ArangoDB.

    :param original_documents: Original documents in the format
                                 {'key': {'any-attr-key': 'any-attr-value'}}.
    :return: List of documents with '_key' attribute and additional attributes.
    """
    transformed_documents = []

    for key, values in original_documents.items():
        transformed_doc = {"_key": key}
        transformed_doc.update(values)
        transformed_documents.append(transformed_doc)

    return transformed_documents


def upsert_collection_documents(db: StandardDatabase, separated: Any) -> Any:
    """
    Process each collection in the separated dictionary.

    :param db: The ArangoDB database object.
    :param separated: A dictionary where the keys are collection names and the
                      values are dictionaries
                      of key-value pairs belonging to those collections.
    :return: A list of results from the insert_many operation.

     If inserting a document fails, the exception is not raised but
     returned as an object in the result list.
    """

    results = []

    for collection_name, documents in separated.items():
        collection = db.collection(collection_name)
        transformed_documents = transform_local_documents_for_adb(documents)
        results.append(
            collection.insert_many(
                transformed_documents, silent=False, overwrite_mode="update"
            )
        )

    return results
