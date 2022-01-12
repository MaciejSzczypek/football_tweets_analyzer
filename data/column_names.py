# region common
CREATED_AT_COLUMN_NAME = "created_at"
LABEL_COLUMN_NAME = "label"
QUERY_FLAG_COLUMN_NAME = "query_flag"
TEXT_COLUMN_NAME = "text"
UNIQUE_ID_COLUMN_NAME = "uniqueID"
USER_NAME_COLUMN_NAME = "user_name"
# endregion common
# region liverpool watford source files

LIV_WAT_COLUMN_NAMES = [
    UNIQUE_ID_COLUMN_NAME,
    TEXT_COLUMN_NAME,
    USER_NAME_COLUMN_NAME,
    CREATED_AT_COLUMN_NAME,
]
# region liverpool watford source files
# region sentiment 140 transfer learning file
SENTIMENT_140_COLUMN_NAMES = [
    LABEL_COLUMN_NAME,
    UNIQUE_ID_COLUMN_NAME,
    CREATED_AT_COLUMN_NAME,
    QUERY_FLAG_COLUMN_NAME,
    USER_NAME_COLUMN_NAME,
    TEXT_COLUMN_NAME,
]
# endregion sentiment 140 transfer learning file
# region auxiliary files
ENGLISH_CLUBS_KEY_COLUMN_NAME = "Key"
ENGLISH_CLUBS_NAME_COLUMN_NAME = "Name"
ENGLISH_CLUBS_CODE_COLUMN_NAME = "Code"
# endregion auxiliary files
