#!/usr/bin/env bash

if [ -z "$1" ] ; then
    echo "No input directory specified"
    exit 1
fi

if [ -z "$2" ] ; then
    echo "No output directory specified"
    exit 1
fi

temp_name=$3
if [ -z "$3" ] ; then
    temp_name="temp.txt"
fi

mkdir -p $2

for file in $(find $1 -type f -name '*.pcap')
do
    filename=$(basename "$file")
    filename_without_ext="${filename%.*}"
    
    # Use tshark to read pcap and write to a temporary text file
    tshark -r $file > data/tmp/$temp_name

    # Process temporary file and save it to output dir
    python3 preprocess.py --file=data/tmp/$temp_name --dir=$2 --label=$filename_without_ext
done