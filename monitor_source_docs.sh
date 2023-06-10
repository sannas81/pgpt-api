#!/bin/bash

DIR_NAME="./source_documents"
DB_DIR_NAME="./db"
DIGEST_FILE="source_docs.digest"
PREV_DIGEST="string"
CURR_DIGEST="string"

if [ -f $DIGEST_FILE ]; then
	PREV_DIGEST=`cat $DIGEST_FILE`
se
	touch $DIGEST_FILE
fi
CURR_DIGEST=`ls -l $DIR_NAME | md5sum`
echo $CURR_DIGEST

if [[ $CURR_DIGEST != $PREV_DIGEST ]]; then
	rm -r $DB_DIR_NAME
	python3 ingest.py
fi
